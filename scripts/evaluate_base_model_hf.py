import os
import sys
import json
import torch
import logging
import argparse
from tqdm import tqdm
from PIL import Image
from typing import Dict, List, Tuple
from datasets import load_dataset

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="评估基础多模态模型")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="模型名称或路径")
    parser.add_argument("--output_file", type=str, default="base_model_results.json", help="结果输出文件")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数，用于快速测试")
    parser.add_argument("--verbose", action="store_true", help="是否输出详细日志")
    
    return parser.parse_args()

def evaluate_base_model(model, processor, benchmark_data, batch_size, device, max_samples=None, verbose=False):
    """评估基础模型在VLRewardBench上的表现"""
    model.eval()
    results = []
    
    # 如果设置了最大样本数，则截取数据集
    if max_samples is not None:
        # 正确处理数据集切片
        benchmark_data = benchmark_data.select(range(min(max_samples, len(benchmark_data))))
    
    # 打印第一个样本的结构，以便了解字段名称
    logger.info(f"数据集样本结构: {benchmark_data[0].keys()}")
    
    # 按批次处理数据
    total_samples = len(benchmark_data)
    for i in tqdm(range(0, total_samples, batch_size), desc="Evaluating"):
        # 正确获取批次数据
        end_idx = min(i + batch_size, total_samples)
        batch_indices = list(range(i, end_idx))
        batch = benchmark_data.select(batch_indices)
        
        # 准备query和response的输入
        for item_idx, item in enumerate(batch):
            # 获取查询和回答
            query = item["query"]
            responses = item["response"]
            image = item["image"]
            human_ranking = item["human_ranking"]
            
            if verbose:
                logger.info(f"处理样本 {item.get('id', f'example_{batch_indices[item_idx]}')}:")
                logger.info(f"查询: {query}")
                logger.info(f"回答1: {responses[0][:100]}...")
                logger.info(f"回答2: {responses[1][:100]}...")
                logger.info(f"人类偏好排名: {human_ranking}")
            
            # 处理每个回答
            all_scores = []
            for resp_idx, resp in enumerate(responses):
                # 组合查询和回答
                full_text = f"{query}\n{resp}"
                
                if verbose:
                    logger.info(f"处理回答 {resp_idx+1}:")
                    logger.info(f"完整文本: {full_text[:100]}...")
                
                # 使用processor处理文本和图像
                inputs = processor(text=full_text, images=image, return_tensors="pt").to(device)
                
                # 打印inputs的键，以便了解正确的参数名称
                if i == 0 and item_idx == 0 and resp_idx == 0:
                    logger.info(f"处理器输出的键: {inputs.keys()}")
                    if verbose:
                        logger.info(f"输入形状:")
                        for k, v in inputs.items():
                            if isinstance(v, torch.Tensor):
                                logger.info(f"  {k}: {v.shape}")
                
                # 获取模型预测
                with torch.no_grad():
                    # 根据文档，Qwen2VL模型只接受input_ids和attention_mask参数
                    model_inputs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"],
                    }
                    
                    # 调用模型
                    outputs = model(**model_inputs, output_hidden_states=True)
                    last_hidden_states = outputs.hidden_states[-1]
                    
                    if verbose:
                        logger.info(f"最后一层隐藏状态形状: {last_hidden_states.shape}")
                    
                    # 获取序列中最后一个token的隐藏状态
                    sequence_lengths = torch.sum(inputs.attention_mask, dim=1) - 1
                    batch_size = last_hidden_states.shape[0]
                    
                    if verbose:
                        logger.info(f"序列长度: {sequence_lengths}")
                    
                    # 收集每个序列的最后一个token的隐藏状态
                    last_token_hidden_states = last_hidden_states[
                        torch.arange(batch_size, device=last_hidden_states.device),
                        sequence_lengths
                    ]
                    
                    if verbose:
                        logger.info(f"最后一个token的隐藏状态形状: {last_token_hidden_states.shape}")
                    
                    # 使用隐藏状态的范数作为分数
                    score = torch.norm(last_token_hidden_states, dim=1).item()
                    all_scores.append(score)
                    
                    if verbose:
                        logger.info(f"回答 {resp_idx+1} 的分数: {score}")
            
            # 确定模型的偏好
            model_preferred_idx = 0 if all_scores[0] > all_scores[1] else 1
            
            # 确定人类偏好
            # 在VLRewardBench中，human_ranking为[0, 1]表示第一个回答更好
            # 为[1, 0]表示第二个回答更好
            human_preferred_idx = 0 if human_ranking[0] < human_ranking[1] else 1
            
            # 检查模型是否与人类偏好一致
            is_correct = model_preferred_idx == human_preferred_idx
            
            if verbose:
                logger.info(f"所有分数: {all_scores}")
                logger.info(f"模型偏好: 回答 {model_preferred_idx+1}")
                logger.info(f"人类偏好: 回答 {human_preferred_idx+1}")
                logger.info(f"模型预测是否正确: {is_correct}")
                logger.info("-" * 50)
            
            # 记录结果
            results.append({
                "id": item.get("id", f"example_{batch_indices[item_idx]}"),
                "scores": all_scores,
                "model_preferred_idx": model_preferred_idx,
                "human_preferred_idx": human_preferred_idx,
                "is_correct": is_correct
            })
    
    # 计算总体准确率
    accuracy = sum(item["is_correct"] for item in results) / len(results)
    
    return {
        "accuracy": accuracy,
        "details": results
    }

def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    logger.info(f"加载模型: {args.model_name_or_path}")
    
    # 使用特定的模型类而不是AutoModelForCausalLM
    if "qwen" in args.model_name_or_path.lower():
        from transformers import Qwen2VLModel, Qwen2VLProcessor
        
        model = Qwen2VLModel.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        processor = Qwen2VLProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    else:
        from transformers import AutoModel, AutoProcessor
        
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    # 加载VLRewardBench数据集
    logger.info("加载VLRewardBench数据集")
    dataset = load_dataset("MMInstruction/VL-RewardBench")
    benchmark_data = dataset["test"]
    
    # 创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # 评估模型
    logger.info("开始评估模型")
    results = evaluate_base_model(
        model, 
        processor, 
        benchmark_data, 
        args.batch_size, 
        device,
        args.max_samples,
        args.verbose
    )
    
    # 保存结果
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估完成，准确率: {results['accuracy']:.4f}")
    logger.info(f"结果已保存到 {args.output_file}")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    main() 