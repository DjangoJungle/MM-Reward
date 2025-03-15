#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:$PYTHONPATH"

# 使用conda环境中的Python
PYTHON=/vol3/ctr/.conda/envs/mllm/bin/python

# 设置模型和输出路径
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
OUTPUT_DIR="./outputs/qwen2-vl-2b-instruct-mmpr-reward-full"
LOG_FILE="./train_mmpr_full.log"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 训练参数 - 使用完整数据集的参数
BATCH_SIZE=1
GRAD_ACCUM=16
EPOCHS=3
LR=5e-6
MAX_SEQ_LEN=2048
LOGGING_STEPS=50
EVAL_STEPS=500
SAVE_STEPS=1000

# 使用LoRA进行参数高效微调
USE_LORA=true
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# 设置wandb项目和运行名称
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WANDB_PROJECT="vlreward"
WANDB_NAME="qwen2-vl-2b-mmpr-full-${TIMESTAMP}"
WANDB_WATCH="gradients"  # 监控梯度
WANDB_LOG_STEPS=50  # 每50步记录一次

# 确保wandb已登录
echo "确保wandb已登录..."
$PYTHON -c "import wandb; print(f'当前登录用户: {wandb.api.viewer()}')"

# 运行训练脚本
echo "开始训练完整数据集..."
echo "训练日志将保存到 $LOG_FILE"
echo "训练过程可在wandb项目 $WANDB_PROJECT 中查看，运行名称为 $WANDB_NAME"

# 使用nohup在后台运行，以防止SSH断开连接导致训练中断
nohup $PYTHON -m train.train \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --max_seq_length $MAX_SEQ_LEN \
    --fp16 \
    --use_mmpr_dataset \
    --validation_split_ratio 0.05 \
    --logging_steps $LOGGING_STEPS \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --save_total_limit 3 \
    --use_lora \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --remove_unused_columns False \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --wandb_name $WANDB_NAME \
    --wandb_watch $WANDB_WATCH \
    --wandb_log_steps $WANDB_LOG_STEPS > $LOG_FILE 2>&1 &

# 保存进程ID，以便稍后检查
PID=$!
echo "训练进程已启动，PID: $PID"
echo "可以使用以下命令查看训练进度："
echo "  tail -f $LOG_FILE"
echo "  ps -p $PID -o pid,ppid,cmd,%cpu,%mem,etime"
echo "  nvidia-smi"

# 创建一个监控脚本
cat > monitor_training.sh << EOF
#!/bin/bash

# 监控训练进程
echo "监控训练进程 PID: $PID"
while ps -p $PID > /dev/null; do
    echo "============ $(date) ============"
    echo "进程状态:"
    ps -p $PID -o pid,ppid,cmd,%cpu,%mem,etime
    echo ""
    echo "GPU状态:"
    nvidia-smi
    echo ""
    echo "最新日志:"
    tail -n 20 $LOG_FILE
    echo ""
    echo "按Ctrl+C退出监控"
    sleep 60
done

echo "训练进程已结束"
EOF

chmod +x monitor_training.sh
echo "可以使用 ./monitor_training.sh 命令监控训练进程" 