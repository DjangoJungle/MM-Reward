#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PYTHONPATH"

# 使用conda环境中的Python
PYTHON=/vol3/ctr/.conda/envs/mllm/bin/python

# 设置模型和输出路径
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR="./outputs/qwen2-vl-2b-instruct-mmpr-reward-test"
LOG_FILE="./train_mmpr_test.log"

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
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WANDB_PROJECT="vlreward"
WANDB_NAME="qwen2-vl-2b-mmpr-test-${TIMESTAMP}"

# 确保wandb已登录
echo "确保wandb已登录..."
$PYTHON -c "import wandb; print(f'当前登录用户: {wandb.api.viewer()}')"

# 运行训练脚本
echo "开始训练..."
$PYTHON -m train.train \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --max_seq_length $MAX_SEQ_LEN \
    --fp16 \
    --use_mmpr_dataset \
    --validation_split_ratio 0.1 \
    --logging_steps 5 \
    --eval_steps 20 \
    --save_steps 50 \
    --save_total_limit 1 \
    --use_lora \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --max_train_samples 100 \
    --remove_unused_columns False \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --wandb_name $WANDB_NAME

echo "测试训练完成，模型已保存到 $OUTPUT_DIR"
echo "训练过程可在wandb项目 $WANDB_PROJECT 中查看，运行名称为 $WANDB_NAME" 