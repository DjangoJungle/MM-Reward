#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import wandb
import logging
import argparse
from tqdm import tqdm
from PIL import Image
from typing import Dict, List, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen2VLProcessor, Qwen2VLModel

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="评估基础模型")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="基础模型路径")
    parser.add_argument("--output_file", type=str, default="./results/base_model_evaluation_results.json", help="结果输出文件")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--max_samples", type=int, default=1000, help="最大样本数，设为None评估全部样本")
    parser.add_argument("--wandb_project", type=str, default="vlreward-eval", help="wandb项目名称")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb运行名称")
    parser.add_argument("--verbose", action="store_true", help="是否输出详细日志")
    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔（样本数）")
    
    return parser.parse_args()

class BaseRewardModel(torch.nn.Module):
    """基础奖励模型，使用原始的Qwen2-VL模型"""
    def __init__(self, model_path):
        super().__init__()
        
        # 加载基础模型
        logger.info(f"加载基础模型: {model_path}")
        self.base_model = Qwen2VLModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # 创建评分头
        hidden_size = self.base_model.config.hidden_size
        logger.info(f"创建评分头，隐藏层大小: {hidden_size}")
        self.score_head = torch.nn.Linear(hidden_size, 1)
        
        # 初始化评分头权重
        torch.nn.init.normal_(self.score_head.weight, mean=0.0, std=0.02)
        
        # 尝试加载LoRA模型中的score_head.pt文件
        score_head_path = "./outputs/score_head.pt"
        if os.path.exists(score_head_path):
            logger.info(f"加载score_head权重: {score_head_path}")
            self.score_head.load_state_dict(torch.load(score_head_path, map_location="cpu"))
        else:
            logger.warning(f"未找到score_head权重文件: {score_head_path}，使用随机初始化的评分头")
        
        # 将score_head移动到与base_model相同的设备
        self.score_head = self.score_head.to(self.base_model.device).to(self.base_model.dtype)
    
    def forward(self, input_ids, attention_mask, pixel_values=None):
        """前向传播函数"""
        # 获取基础模型的输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # 获取最后一层隐藏状态
        last_hidden_states = outputs.hidden_states[-1]
        
        # 获取序列中最后一个token的隐藏状态
        sequence_lengths = torch.sum(attention_mask, dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        
        # 收集每个序列的最后一个token的隐藏状态
        last_token_hidden_states = last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]
        
        # 通过scorehead计算奖励分数
        scores = self.score_head(last_token_hidden_states).squeeze(-1)
        
        return scores

def evaluate_model(args):
    # 初始化wandb
    if args.wandb_name:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            tags=["reward-model", "evaluation", "base-model"],
            notes=f"评估基础模型: {args.model_path}"
        )
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaseRewardModel(args.model_path)
    model.eval()
    
    # 记录模型信息
    if args.wandb_name and hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        wandb.config.update({"model_config": model.base_model.config.to_dict()}, allow_val_change=True)
    
    # 加载tokenizer和processor
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = Qwen2VLProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 加载测试数据集
    logger.info("加载测试数据集")
    try:
        # 尝试加载VL-RewardBench数据集
        dataset = load_dataset("MMInstruction/VL-RewardBench", split="test")
        logger.info(f"成功加载VL-RewardBench数据集，包含 {len(dataset)} 个样本")
    except Exception as e:
        logger.warning(f"加载VL-RewardBench数据集失败: {e}，使用MMPR数据集的验证集")
        # 使用MMPR数据集的验证集
        full_dataset = load_dataset("YennNing/MMPR_combined_dataset")
        train_val = full_dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = train_val["test"]
        logger.info(f"使用MMPR数据集的验证集，包含 {len(dataset)} 个样本")
    
    # 限制样本数
    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        logger.info(f"限制样本数为 {args.max_samples}")
    
    # 记录数据集信息
    if args.wandb_name:
        wandb.config.update({
            "dataset_size": len(dataset),
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
        }, allow_val_change=True)
    
    # 评估结果
    results = []
    correct_count = 0
    total_count = 0
    
    # 按批次处理数据
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="评估进度"):
        batch_indices = list(range(i, min(i + args.batch_size, len(dataset))))
        batch = dataset.select(batch_indices)
        
        for item_idx, item in enumerate(batch):
            try:
                # 获取查询和回答
                if "query" in item and "response" in item:
                    # VL-RewardBench格式
                    query = item["query"]
                    responses = item["response"]
                    image = item["image"]
                    human_ranking = item["human_ranking"]
                    
                    # 处理图像
                    image_processed = processor(text="", images=image, return_tensors="pt").pixel_values.to(device)
                    
                    # 处理每个回答
                    all_scores = []
                    for resp_idx, resp in enumerate(responses):
                        # 组合查询和回答
                        full_text = f"{query}\n{resp}"
                        
                        # 使用tokenizer处理文本
                        tokens = tokenizer(
                            full_text,
                            max_length=2048,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
                        ).to(device)
                        
                        # 获取模型预测
                        with torch.no_grad():
                            score = model(
                                input_ids=tokens.input_ids,
                                attention_mask=tokens.attention_mask,
                                pixel_values=image_processed
                            ).item()
                            
                            all_scores.append(score)
                    
                    # 确定模型的偏好
                    model_preferred_idx = 0 if all_scores[0] > all_scores[1] else 1
                    
                    # 确定人类偏好
                    human_preferred_idx = 0 if human_ranking[0] < human_ranking[1] else 1
                    
                    # 检查模型是否与人类偏好一致
                    is_correct = model_preferred_idx == human_preferred_idx
                    if is_correct:
                        correct_count += 1
                    total_count += 1
                    
                    # 记录结果
                    result = {
                        "id": item.get("id", f"example_{batch_indices[item_idx]}"),
                        "scores": all_scores,
                        "model_preferred_idx": model_preferred_idx,
                        "human_preferred_idx": human_preferred_idx,
                        "is_correct": is_correct,
                        "query": query,
                        "responses": responses,
                    }
                    results.append(result)
                    
                    # 记录到wandb
                    if args.wandb_name and (total_count % args.log_interval == 0 or total_count == 1):
                        wandb.log({
                            "sample_accuracy": 1.0 if is_correct else 0.0,
                            "score_diff": abs(all_scores[0] - all_scores[1]),
                            "chosen_score": all_scores[human_preferred_idx],
                            "rejected_score": all_scores[1 - human_preferred_idx],
                            "running_accuracy": correct_count / total_count,
                            "samples_evaluated": total_count,
                        })
                    
                    if args.verbose and (total_count % args.log_interval == 0 or total_count == 1):
                        logger.info(f"样本 {result['id']} - 模型分数: {all_scores}, 模型偏好: {model_preferred_idx}, 人类偏好: {human_preferred_idx}, 正确: {is_correct}, 当前准确率: {correct_count}/{total_count} = {correct_count/total_count:.4f}")
                
                elif "question" in item and "chosen" in item and "rejected" in item:
                    # MMPR数据集格式
                    question = item["question"]
                    chosen = item["chosen"]
                    rejected = item["rejected"]
                    image = item["image"]
                    
                    # 处理图像
                    image_processed = processor(text="", images=image, return_tensors="pt").pixel_values.to(device)
                    
                    # 处理chosen文本
                    chosen_text = f"{question}\n{chosen}"
                    chosen_tokens = tokenizer(
                        chosen_text,
                        max_length=2048,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    # 处理rejected文本
                    rejected_text = f"{question}\n{rejected}"
                    rejected_tokens = tokenizer(
                        rejected_text,
                        max_length=2048,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    # 获取模型预测
                    with torch.no_grad():
                        chosen_score = model(
                            input_ids=chosen_tokens.input_ids,
                            attention_mask=chosen_tokens.attention_mask,
                            pixel_values=image_processed
                        ).item()
                        
                        rejected_score = model(
                            input_ids=rejected_tokens.input_ids,
                            attention_mask=rejected_tokens.attention_mask,
                            pixel_values=image_processed
                        ).item()
                    
                    # 检查模型是否正确预测
                    is_correct = chosen_score > rejected_score
                    if is_correct:
                        correct_count += 1
                    total_count += 1
                    
                    # 记录结果
                    result = {
                        "id": item.get("idx", f"example_{batch_indices[item_idx]}"),
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                        "is_correct": is_correct,
                        "question": question,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                    results.append(result)
                    
                    # 记录到wandb
                    if args.wandb_name and (total_count % args.log_interval == 0 or total_count == 1):
                        wandb.log({
                            "sample_accuracy": 1.0 if is_correct else 0.0,
                            "score_diff": chosen_score - rejected_score,
                            "chosen_score": chosen_score,
                            "rejected_score": rejected_score,
                            "running_accuracy": correct_count / total_count,
                            "samples_evaluated": total_count,
                        })
                    
                    if args.verbose and (total_count % args.log_interval == 0 or total_count == 1):
                        logger.info(f"样本 {result['id']} - Chosen分数: {chosen_score}, Rejected分数: {rejected_score}, 正确: {is_correct}, 当前准确率: {correct_count}/{total_count} = {correct_count/total_count:.4f}")
                
                else:
                    logger.warning(f"样本格式不支持: {item.keys()}")
            
            except Exception as e:
                logger.error(f"处理样本 {batch_indices[item_idx]} 时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # 计算总体准确率
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    logger.info(f"总体准确率: {accuracy:.4f} ({correct_count}/{total_count})")
    
    # 记录最终结果
    final_results = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "model_path": args.model_path,
        "details": results
    }
    
    # 保存结果
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果已保存到 {args.output_file}")
    
    # 记录最终指标到wandb
    if args.wandb_name:
        wandb.log({
            "final_accuracy": accuracy,
            "total_samples": total_count,
        })
        
        # 创建准确率表格
        accuracy_table = wandb.Table(columns=["Sample ID", "Is Correct", "Chosen Score", "Rejected Score", "Score Diff"])
        for result in results:
            if "chosen_score" in result and "rejected_score" in result:
                accuracy_table.add_data(
                    result["id"],
                    result["is_correct"],
                    result["chosen_score"],
                    result["rejected_score"],
                    result["chosen_score"] - result["rejected_score"]
                )
            elif "scores" in result:
                accuracy_table.add_data(
                    result["id"],
                    result["is_correct"],
                    result["scores"][result["human_preferred_idx"]],
                    result["scores"][1 - result["human_preferred_idx"]],
                    result["scores"][result["human_preferred_idx"]] - result["scores"][1 - result["human_preferred_idx"]]
                )
        
        wandb.log({"accuracy_table": accuracy_table})
        
        # 完成wandb
        wandb.finish()
    
    return final_results

def main():
    args = parse_args()
    evaluate_model(args)

if __name__ == "__main__":
    main() 