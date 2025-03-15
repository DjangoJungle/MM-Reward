#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PYTHONPATH"

# 使用conda环境中的Python
PYTHON=/vol3/ctr/.conda/envs/mllm/bin/python

# 设置模型和输出路径
MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"  # 基础模型路径
RESULTS_DIR="./results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_FILE="${RESULTS_DIR}/base_model_evaluation_results_${TIMESTAMP}.json"

# 创建结果目录
mkdir -p $RESULTS_DIR

# 设置wandb项目和运行名称
WANDB_PROJECT="vlreward-eval"
WANDB_NAME="qwen2-vl-2b-base-eval-${TIMESTAMP}"

# 确保wandb已登录
echo "确保wandb已登录..."
$PYTHON -c "import wandb; print(f'当前登录用户: {wandb.api.viewer()}')"

# 运行评估脚本
echo "开始评估基础模型..."
$PYTHON evaluate_base_model.py \
    --model_path $MODEL_PATH \
    --output_file $OUTPUT_FILE \
    --batch_size 1 \
    --max_samples 500 \
    --wandb_project $WANDB_PROJECT \
    --wandb_name $WANDB_NAME \
    --verbose \
    --log_interval 10

echo "评估完成，结果已保存到 $OUTPUT_FILE"
echo "评估过程可在wandb项目 $WANDB_PROJECT 中查看，运行名称为 $WANDB_NAME" 