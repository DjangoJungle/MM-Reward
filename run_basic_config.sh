#!/bin/bash
set -e  # 遇到错误立即退出

# 创建目录结构
mkdir -p data/vlrewardbench
mkdir -p models
mkdir -p train
mkdir -p evaluate
mkdir -p scripts
mkdir -p results
mkdir -p outputs

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 第一步：评估基础模型在VLRewardBench上的分数
echo "评估基础模型..."
python scripts/evaluate_base_model_hf.py \
  --model_name_or_path "Qwen/Qwen2-VL-2B-Instruct" \
  --output_file "results/qwen2-vl-2b-instruct-base.json"



echo "所有步骤完成！" 