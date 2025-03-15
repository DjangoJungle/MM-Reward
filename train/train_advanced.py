#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import set_seed, get_reward_model
from data.dataset import PreferenceDataset
from data.collator import PreferenceDataCollator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="高级奖励模型训练脚本")
    
    # 基础参数
    parser.add_argument("--model_name_or_path", type=str, required=True, help="预训练模型名称或路径")
    parser.add_argument("--data_path", type=str, required=True, help="训练数据路径")
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
    parser.add_argument("--temperature", type=float, default=0.5, 
                        help="温度参数（用于contrastive模型）")
    parser.add_argument("--contrastive_weight", type=float, default=0.5, 
                        help="对比损失权重（用于contrastive模型）")
    parser.add_argument("--alpha", type=float, default=1.0, 
                        help="基础损失权重（用于adaptive模型）")
    parser.add_argument("--beta", type=float, default=0.5, 
                        help="难度加权系数（用于adaptive模型）")
    parser.add_argument("--gamma", type=float, default=2.0, 
                        help="聚焦参数（用于adaptive模型）")
    
    # 训练参数
    parser.add_argument("--train_batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500, help="模型保存步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    
    # 其他参数
    parser.add_argument("--fp16", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地排名")
    
    return parser.parse_args()

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
    elif args.model_type == "contrastive":
        model_kwargs["temperature"] = args.temperature
        model_kwargs["contrastive_weight"] = args.contrastive_weight
    elif args.model_type == "fusion":
        model_kwargs["fusion_method"] = args.fusion_method
    elif args.model_type == "adaptive":
        model_kwargs["alpha"] = args.alpha
        model_kwargs["beta"] = args.beta
        model_kwargs["gamma"] = args.gamma
    
    # 加载模型
    logger.info(f"加载{args.model_type}类型的奖励模型: {args.model_name_or_path}")
    model = get_reward_model(
        model_name_or_path=args.model_name_or_path,
        model_type=args.model_type,
        **model_kwargs
    )
    model.to(device)
    
    # 加载数据集
    logger.info(f"加载数据集: {args.data_path}")
    dataset = PreferenceDataset(args.data_path)
    data_collator = PreferenceDataCollator(
        tokenizer=None,  # 模型内部已有tokenizer
        max_length=args.max_seq_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    
    # 准备优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(dataloader) * args.num_train_epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # 训练循环
    logger.info("开始训练...")
    global_step = 0
    total_loss = 0.0
    
    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_train_epochs}")
        
        for step, batch in enumerate(epoch_iterator):
            # 将数据移动到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 前向传播
            with torch.cuda.amp.autocast() if args.fp16 else torch.no_grad():
                chosen_scores = model(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"]
                )
                rejected_scores = model(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"]
                )
                loss = model.compute_preference_loss(chosen_scores, rejected_scores)
                
                # 梯度累积
                loss = loss / args.gradient_accumulation_steps
            
            # 反向传播
            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item()
            
            # 梯度更新
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 日志记录
                if global_step % args.logging_steps == 0:
                    avg_loss = total_loss / args.logging_steps
                    logger.info(f"步骤: {global_step}, 损失: {avg_loss:.4f}")
                    total_loss = 0.0
                
                # 保存模型
                if global_step % args.save_steps == 0:
                    output_path = os.path.join(
                        args.output_dir,
                        f"{args.model_type}_model_step_{global_step}"
                    )
                    os.makedirs(output_path, exist_ok=True)
                    model.save_pretrained(output_path)
                    logger.info(f"模型保存到: {output_path}")
    
    # 保存最终模型
    final_output_path = os.path.join(
        args.output_dir,
        f"{args.model_type}_model_final"
    )
    os.makedirs(final_output_path, exist_ok=True)
    model.save_pretrained(final_output_path)
    logger.info(f"最终模型保存到: {final_output_path}")
    
    logger.info("训练完成!")

if __name__ == "__main__":
    main() 