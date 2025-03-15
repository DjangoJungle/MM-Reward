#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import logging
import argparse
from transformers import Qwen2VLModel

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="创建score_head.pt文件")
    parser.add_argument("--model_path", type=str, default="./outputs", help="模型路径")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="基础模型路径")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载基础模型配置
    logger.info(f"加载基础模型配置: {args.base_model}")
    base_model = Qwen2VLModel.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 获取隐藏层大小
    if hasattr(base_model.config, "hidden_size"):
        hidden_size = base_model.config.hidden_size
    elif hasattr(base_model.config, "d_model"):
        hidden_size = base_model.config.d_model
    else:
        hidden_size = 768  # 默认值
    
    logger.info(f"隐藏层大小: {hidden_size}")
    
    # 创建评分头
    score_head = torch.nn.Linear(hidden_size, 1)
    
    # 初始化评分头权重
    torch.nn.init.normal_(score_head.weight, mean=0.0, std=0.02)
    
    # 保存score_head
    score_head_path = os.path.join(args.model_path, "score_head.pt")
    torch.save(score_head.state_dict(), score_head_path)
    
    logger.info(f"score_head已保存到: {score_head_path}")

if __name__ == "__main__":
    main() 