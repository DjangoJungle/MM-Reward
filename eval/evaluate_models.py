#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import set_seed, get_reward_model
from data.dataset import EvalDataset
from data.collator import EvalDataCollator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="奖励模型评估脚本")
    
    # 基础参数
    parser.add_argument("--model_name_or_path", type=str, required=True, help="模型名称或路径")
    parser.add_argument("--data_path", type=str, required=True, help="评估数据路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    # 模型参数
    parser.add_argument("--model_type", type=str, default="base", 
                        choices=["base", "enhanced", "pooling", "contrastive", "fusion", "adaptive"],
                        help="奖励模型类型")
    
    # 特定模型参数
    parser.add_argument("--pooling_type", type=str, default="mean", 
                        choices=["mean", "max", "last"],
                        help="池化类型（用于pooling模型）")
    parser.add_argument("--fusion_method", type=str, default="attention", 
                        choices=["attention", "weighted", "concat", "gated"],
                        help="融合方法（用于fusion模型）")
    
    # 评估参数
    parser.add_argument("--eval_batch_size", type=int, default=8, help="评估批次大小")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    
    return parser.parse_args()

def evaluate(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    
    total = 0
    correct = 0
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 将数据移动到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 获取分数
            chosen_scores = model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"]
            )
            rejected_scores = model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"]
            )
            
            # 计算准确率
            predictions = (chosen_scores > rejected_scores).float()
            correct += predictions.sum().item()
            total += predictions.size(0)
            
            # 收集预测结果
            for i in range(len(predictions)):
                all_predictions.append({
                    "chosen_score": chosen_scores[i].item(),
                    "rejected_score": rejected_scores[i].item(),
                    "prediction": predictions[i].item(),
                    "sample_id": batch.get("sample_ids", [i])[i]
                })
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "predictions": all_predictions
    }

def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备模型特定参数
    model_kwargs = {}
    if args.model_type == "pooling":
        model_kwargs["pooling_type"] = args.pooling_type
    elif args.model_type == "fusion":
        model_kwargs["fusion_method"] = args.fusion_method
    
    # 加载模型
    logger.info(f"加载{args.model_type}类型的奖励模型: {args.model_name_or_path}")
    model = get_reward_model(
        model_name_or_path=args.model_name_or_path,
        model_type=args.model_type,
        **model_kwargs
    )
    model.to(device)
    
    # 加载数据集
    logger.info(f"加载评估数据集: {args.data_path}")
    dataset = EvalDataset(args.data_path)
    data_collator = EvalDataCollator(
        tokenizer=None,  # 模型内部已有tokenizer
        max_length=args.max_seq_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    
    # 评估模型
    logger.info("开始评估...")
    results = evaluate(model, dataloader, device)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir,
        f"{args.model_type}_eval_results_{timestamp}.json"
    )
    
    with open(output_file, "w") as f:
        json.dump({
            "model_type": args.model_type,
            "model_path": args.model_name_or_path,
            "accuracy": results["accuracy"],
            "predictions": results["predictions"],
            "eval_params": vars(args)
        }, f, indent=2)
    
    logger.info(f"评估完成! 准确率: {results['accuracy']:.4f}")
    logger.info(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 