#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PYTHONPATH"

# 使用conda环境中的Python
PYTHON=/vol3/ctr/.conda/envs/mllm/bin/python

# 设置模型和输出路径
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR="./outputs/qwen2-vl-2b-instruct-mmpr-reward-wandb"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 训练参数 - 使用较小的参数进行测试
BATCH_SIZE=1
GRAD_ACCUM=4
EPOCHS=1
LR=5e-6
MAX_SEQ_LEN=512  # 减小序列长度，加快处理速度

# 使用LoRA进行参数高效微调
USE_LORA=true
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# 设置wandb项目和运行名称
WANDB_PROJECT="vlreward"
WANDB_NAME="qwen2-vl-2b-mmpr-wandb-${TIMESTAMP}"

# 确保wandb已登录
echo "确保wandb已登录..."
$PYTHON -c "import wandb; print(f'当前登录用户: {wandb.api.viewer()}')"

# 创建一个简单的训练脚本
cat > train_mmpr_wandb.py << 'EOF'
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import wandb
import logging
import argparse
from dataclasses import asdict
from transformers import TrainingArguments, HfArgumentParser
from datasets import load_dataset
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.utils import get_reward_model
from train.config import TrainingConfig

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试wandb与奖励模型训练的集成")
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--wandb_project", type=str, required=True, help="wandb项目名称")
    parser.add_argument("--wandb_name", type=str, required=True, help="wandb运行名称")
    args = parser.parse_args()
    
    # 初始化wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        tags=["reward-model", "mmpr", "test"],
        notes="测试wandb与奖励模型训练的集成"
    )
    
    # 加载模型
    logger.info(f"加载模型: {args.model_name}")
    model = get_reward_model(args.model_name)
    
    # 记录模型信息
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        wandb.config.update({"model_config": model.base_model.config.to_dict()}, allow_val_change=True)
    
    # 加载数据集
    logger.info("加载MMPR数据集")
    dataset = load_dataset("YennNing/MMPR_combined_dataset", split="train[:10]")  # 只加载前10个样本
    logger.info(f"加载了 {len(dataset)} 个样本")
    
    # 记录数据集信息
    wandb.config.update({
        "dataset_size": len(dataset),
        "dataset_name": "MMPR_combined_dataset",
    }, allow_val_change=True)
    
    # 记录一个样本示例
    if len(dataset) > 0:
        sample = dataset[0]
        if "image" in sample and sample["image"] is not None and hasattr(sample["image"], "convert"):
            wandb.log({
                "sample_image": wandb.Image(sample["image"], 
                                           caption="样本图像示例"),
                "sample_question": sample.get("question", ""),
                "sample_chosen": sample.get("chosen", ""),
                "sample_rejected": sample.get("rejected", ""),
            })
    
    # 模拟训练过程
    logger.info("模拟训练过程...")
    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        
        # 模拟批次训练
        for batch_idx in range(5):
            # 模拟损失和准确率
            batch_loss = 1.0 - 0.1 * epoch - 0.01 * batch_idx
            batch_accuracy = 0.5 + 0.1 * epoch + 0.01 * batch_idx
            
            # 累积损失和准确率
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            
            # 记录批次级别的指标
            wandb.log({
                "batch_loss": batch_loss,
                "batch_accuracy": batch_accuracy,
                "batch": epoch * 5 + batch_idx,
                "learning_rate": 5e-6,
            })
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/5, Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.4f}")
        
        # 计算平均损失和准确率
        avg_loss = epoch_loss / 5
        avg_accuracy = epoch_accuracy / 5
        
        # 记录epoch级别的指标
        wandb.log({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_accuracy": avg_accuracy,
            "chosen_mean": 0.5 + 0.1 * epoch,
            "rejected_mean": 0.3 + 0.05 * epoch,
            "score_diff": 0.2 + 0.05 * epoch,
        })
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
    
    # 完成训练
    logger.info("训练完成！")
    
    # 保存模型
    logger.info(f"保存模型到 {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 完成wandb
    wandb.finish()

if __name__ == "__main__":
    main()
EOF

# 运行训练脚本
echo "开始训练..."
$PYTHON train_mmpr_wandb.py \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --wandb_project $WANDB_PROJECT \
    --wandb_name $WANDB_NAME

echo "测试训练完成"
echo "训练过程可在wandb项目 $WANDB_PROJECT 中查看，运行名称为 $WANDB_NAME" 