from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class TrainingConfig:
    """训练配置类"""
    # 模型参数
    model_name_or_path: str = "Qwen/Qwen2-VL-2B-Instruct"  # 使用Qwen2-VL-2B-Instruct
    use_flash_attention: bool = True
    
    # 训练参数
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # 优化器参数
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # 数据参数
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    validation_split_ratio: float = 0.05  # 验证集分割比例
    max_seq_length: int = 2048
    max_train_samples: Optional[int] = None  # 最大训练样本数，用于快速测试
    remove_unused_columns: bool = True  # 是否移除未使用的列
    
    # 其他参数
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # 混合精度训练
    fp16: bool = True
    bf16: bool = False
    
    # 分布式训练
    local_rank: int = -1
    
    # 是否使用wandb记录
    use_wandb: bool = False
    wandb_project: str = "vlreward"
    wandb_name: Optional[str] = None
    wandb_entity: Optional[str] = None  # wandb实体/组织
    wandb_log_model: bool = False  # 是否将模型上传到wandb
    wandb_watch: str = "gradients"  # 可选值: "gradients", "parameters", "all", "false"
    wandb_log_steps: int = 1  # 每多少步记录一次wandb日志 