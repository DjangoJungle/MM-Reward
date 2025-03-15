import os
import sys
import torch
import logging
import argparse
from dataclasses import asdict
from transformers import (
    Trainer, 
    TrainingArguments, 
    HfArgumentParser,
    set_seed as transformers_set_seed
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import get_reward_model, set_seed
from .config import TrainingConfig

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="训练多模态奖励模型")
    parser.add_argument("--config_file", type=str, default=None, help="配置文件路径")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="基础模型名称或路径")
    parser.add_argument("--train_file", type=str, default=None, help="训练数据文件")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA进行参数高效微调")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_mmpr_dataset", action="store_true", help="是否使用MMPR数据集")
    
    args, remaining = parser.parse_known_args()
    
    # 如果提供了配置文件，从文件加载配置
    if args.config_file:
        parser = HfArgumentParser(TrainingConfig)
        config = parser.parse_json_file(args.config_file)[0]
    else:
        # 否则使用默认配置
        config = TrainingConfig()
    
    # 命令行参数覆盖配置文件
    for key, value in vars(args).items():
        if value is not None and key != "config_file" and key != "use_lora" and key != "use_mmpr_dataset" and hasattr(config, key):
            setattr(config, key, value)
    
    # 解析剩余参数
    if remaining:
        parser = HfArgumentParser(TrainingConfig)
        remaining_config = parser.parse_args_into_dataclasses(args=remaining)[0]
        for key, value in asdict(remaining_config).items():
            if value is not None:
                setattr(config, key, value)
    
    return config, args.use_lora, args.lora_r, args.lora_alpha, args.lora_dropout, args.use_mmpr_dataset

def prepare_dataset(config, use_mmpr_dataset=False):
    """准备数据集"""
    if use_mmpr_dataset:
        logger.info("加载MMPR数据集")
        try:
            dataset = load_dataset("YennNing/MMPR_combined_dataset")
            logger.info(f"成功加载MMPR数据集，包含 {len(dataset['train'])} 个训练样本")
            
            # 打印数据集的结构
            logger.info(f"数据集样本结构: {dataset['train'][0].keys()}")
            
            # 如果需要，可以创建一个小的验证集
            if not config.validation_file and config.validation_split_ratio > 0:
                train_val = dataset["train"].train_test_split(
                    test_size=config.validation_split_ratio, 
                    seed=config.seed
                )
                dataset["train"] = train_val["train"]
                dataset["validation"] = train_val["test"]
                logger.info(f"创建验证集，包含 {len(dataset['validation'])} 个样本")
            
            return dataset
        except Exception as e:
            logger.error(f"加载MMPR数据集失败: {e}")
            raise
    elif config.train_file is None:
        raise ValueError("需要提供训练数据文件或使用MMPR数据集")
    else:
        # 加载数据集
        dataset = load_dataset("json", data_files={"train": config.train_file})
        
        # 如果有验证集
        if config.validation_file:
            eval_dataset = load_dataset("json", data_files={"validation": config.validation_file})
            dataset["validation"] = eval_dataset["validation"]
        
        return dataset

def load_image(image_path):
    """加载图像，支持本地路径、URL和PIL图像对象"""
    try:
        # 如果已经是PIL图像对象，直接返回
        if isinstance(image_path, Image.Image):
            return image_path
        
        # 如果是字符串路径
        if isinstance(image_path, str):
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_path).convert('RGB')
            return image
        
        # 如果是其他类型，尝试转换为字符串
        return load_image(str(image_path))
    except Exception as e:
        logger.error(f"加载图像失败: {e}")
        # 返回一个空白图像作为备用
        return Image.new('RGB', (224, 224), color='white')

def prepare_mmpr_dataset(examples, tokenizer, image_processor, max_length=512):
    """
    准备MMPR数据集的函数
    
    输入格式:
    {
        "image": PIL图像对象,
        "question": 输入查询,
        "chosen": 选择的回答,
        "rejected": 拒绝的回答
    }
    """
    batch_size = len(examples["image"])
    
    # 准备输入
    chosen_input_ids = []
    chosen_attention_mask = []
    chosen_pixel_values = []
    
    rejected_input_ids = []
    rejected_attention_mask = []
    rejected_pixel_values = []
    
    for i in range(batch_size):
        try:
            # 确保问题和回答不为None
            question = examples['question'][i] if examples['question'][i] is not None else ""
            chosen_answer = examples['chosen'][i] if examples['chosen'][i] is not None else ""
            rejected_answer = examples['rejected'][i] if examples['rejected'][i] is not None else ""
            
            # 构建chosen文本
            chosen_text = f"{question}\n{chosen_answer}"
            
            # 处理chosen文本
            chosen_tokens = tokenizer(
                chosen_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 构建rejected文本
            rejected_text = f"{question}\n{rejected_answer}"
            
            # 处理rejected文本
            rejected_tokens = tokenizer(
                rejected_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 处理图像 - 首先检查图像是否为None
            image = examples["image"][i]
            if image is None:
                logger.warning(f"样本 {i} 的图像为None，跳过")
                continue
            
            # 确保图像是PIL图像对象
            if not hasattr(image, 'convert'):
                logger.warning(f"样本 {i} 的图像不是PIL图像对象，跳过")
                continue
                
            # 转换为RGB模式
            image = image.convert('RGB')
            
            # 使用image_processor处理图像
            try:
                # 确保图像尺寸至少为28x28像素
                w, h = image.size
                if w < 28 or h < 28:
                    # 计算新尺寸，保持宽高比
                    if w < h:
                        new_w = 28
                        new_h = int(h * (28/w))
                    else:
                        new_h = 28
                        new_w = int(w * (28/h))
                    image = image.resize((new_w, new_h), resample=Image.LANCZOS)
                
                # 正确调用image_processor - 使用text参数
                processed_image = image_processor(
                    text="",  # 添加空文本参数
                    images=image, 
                    return_tensors="pt"
                )
                image_tensor = processed_image.pixel_values[0]
            except Exception as e:
                logger.warning(f"使用image_processor处理图像失败: {e}，使用备用方法")
                # 备用方法：调整图像大小并手动转换为张量
                image = image.resize((224, 224))
                # 转换为张量 [3, 224, 224]
                image_tensor = torch.tensor(list(image.getdata())).reshape(3, 224, 224).float() / 255.0
            
            # 添加到batch
            chosen_input_ids.append(chosen_tokens.input_ids[0])
            chosen_attention_mask.append(chosen_tokens.attention_mask[0])
            chosen_pixel_values.append(image_tensor)
            
            rejected_input_ids.append(rejected_tokens.input_ids[0])
            rejected_attention_mask.append(rejected_tokens.attention_mask[0])
            rejected_pixel_values.append(image_tensor)
        except Exception as e:
            logger.error(f"处理样本 {i} 时出错: {e}")
            # 跳过这个样本
            continue
    
    # 如果所有样本都处理失败，返回空字典
    if len(chosen_input_ids) == 0:
        logger.error("所有样本处理失败")
        return {}
    
    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "chosen_pixel_values": chosen_pixel_values,
        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
        "rejected_pixel_values": rejected_pixel_values,
    }

def prepare_reward_training_dataset(examples, tokenizer, image_processor, max_length):
    """
    准备奖励模型训练数据
    假设数据格式为：
    {
        "chosen": {"text": "...", "image": "..."},
        "rejected": {"text": "...", "image": "..."}
    }
    """
    batch_size = len(examples["chosen"])
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "pixel_values": [],
        "labels": []
    }
    
    for i in range(batch_size):
        # 处理chosen样本
        chosen_text = examples["chosen"][i]["text"]
        chosen_image = examples["chosen"][i]["image"]
        
        # 处理图像
        chosen_image_processed = image_processor(chosen_image).pixel_values[0]
        
        # 处理文本
        chosen_tokens = tokenizer(
            chosen_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 处理rejected样本
        rejected_text = examples["rejected"][i]["text"]
        rejected_image = examples["rejected"][i]["image"]
        
        # 处理图像
        rejected_image_processed = image_processor(rejected_image).pixel_values[0]
        
        # 处理文本
        rejected_tokens = tokenizer(
            rejected_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 添加到batch
        model_inputs["input_ids"].extend([chosen_tokens.input_ids[0], rejected_tokens.input_ids[0]])
        model_inputs["attention_mask"].extend([chosen_tokens.attention_mask[0], rejected_tokens.attention_mask[0]])
        model_inputs["pixel_values"].extend([chosen_image_processed, rejected_image_processed])
        
        # 标签：chosen=1, rejected=0
        model_inputs["labels"].extend([1, 0])
    
    return model_inputs

class PreferenceTrainer(Trainer):
    """
    偏好学习训练器
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算偏好学习损失
        """
        # 确保输入数据类型正确
        chosen_input_ids = inputs["chosen_input_ids"].long()
        chosen_attention_mask = inputs["chosen_attention_mask"].long()
        chosen_pixel_values = inputs["chosen_pixel_values"].to(torch.float16)  # 直接使用float16
        
        rejected_input_ids = inputs["rejected_input_ids"].long()
        rejected_attention_mask = inputs["rejected_attention_mask"].long()
        rejected_pixel_values = inputs["rejected_pixel_values"].to(torch.float16)  # 直接使用float16
        
        # 获取成对分数
        chosen_scores = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            pixel_values=chosen_pixel_values
        )
        
        rejected_scores = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
            pixel_values=rejected_pixel_values
        )
        
        # 计算偏好损失
        loss = model.compute_preference_loss(chosen_scores, rejected_scores)
        
        # 计算准确率
        accuracy = (chosen_scores > rejected_scores).float().mean()
        
        # 计算分数差异
        score_diff = (chosen_scores - rejected_scores).mean().item()
        score_diff_abs = (chosen_scores - rejected_scores).abs().mean().item()
        
        # 计算分数统计信息
        chosen_mean = chosen_scores.mean().item()
        chosen_std = chosen_scores.std().item() if chosen_scores.numel() > 1 else 0.0
        rejected_mean = rejected_scores.mean().item()
        rejected_std = rejected_scores.std().item() if rejected_scores.numel() > 1 else 0.0
        
        # 记录指标
        self.log({
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "score_diff": score_diff,
            "score_diff_abs": score_diff_abs,
            "chosen_mean": chosen_mean,
            "chosen_std": chosen_std,
            "rejected_mean": rejected_mean,
            "rejected_std": rejected_std,
            "chosen_min": chosen_scores.min().item(),
            "chosen_max": chosen_scores.max().item(),
            "rejected_min": rejected_scores.min().item(),
            "rejected_max": rejected_scores.max().item(),
        })
        
        if return_outputs:
            return loss, {"chosen_scores": chosen_scores, "rejected_scores": rejected_scores}
        return loss
        
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """
        预测步骤
        """
        with torch.no_grad():
            # 确保输入数据类型正确
            chosen_input_ids = inputs["chosen_input_ids"].long()
            chosen_attention_mask = inputs["chosen_attention_mask"].long()
            chosen_pixel_values = inputs["chosen_pixel_values"].to(torch.float16)  # 直接使用float16
            
            rejected_input_ids = inputs["rejected_input_ids"].long()
            rejected_attention_mask = inputs["rejected_attention_mask"].long()
            rejected_pixel_values = inputs["rejected_pixel_values"].to(torch.float16)  # 直接使用float16
            
            # 获取成对分数
            chosen_scores = model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                pixel_values=chosen_pixel_values
            )
            
            rejected_scores = model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                pixel_values=rejected_pixel_values
            )
            
            # 计算偏好损失
            loss = model.compute_preference_loss(chosen_scores, rejected_scores)
            
        return (loss, None, None)

def compute_metrics(eval_preds):
    """计算评估指标"""
    chosen_scores, rejected_scores = eval_preds
    # 计算准确率
    accuracy = (chosen_scores > rejected_scores).astype(float).mean()
    return {"accuracy": accuracy}

def main():
    # 解析参数
    config, use_lora, lora_r, lora_alpha, lora_dropout, use_mmpr_dataset = parse_args()
    
    # 设置随机种子
    set_seed(config.seed)
    transformers_set_seed(config.seed)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 加载模型和tokenizer
    model = get_reward_model(config.model_name_or_path)
    
    # 如果使用LoRA
    if use_lora:
        logger.info("使用LoRA进行参数高效微调")
        
        # 获取模型中的所有模块名称
        model_modules = [name for name, _ in model.base_model.named_modules()]
        logger.info(f"模型中的模块: {model_modules[:10]}...")
        
        # 为Qwen2VL模型设置合适的target_modules
        if "qwen" in config.model_name_or_path.lower():
            # 根据日志中显示的实际模块名称
            target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",  # 修改为FEATURE_EXTRACTION而不是CAUSAL_LM
            target_modules=target_modules
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # 获取tokenizer和image_processor
    if "qwen" in config.model_name_or_path.lower():
        from transformers import AutoTokenizer, Qwen2VLProcessor
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        image_processor = Qwen2VLProcessor.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    elif "internvl" in config.model_name_or_path.lower():
        from transformers import AutoTokenizer, AutoProcessor
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        image_processor = AutoProcessor.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    else:
        raise ValueError(f"不支持的模型: {config.model_name_or_path}")
    
    # 准备数据集
    dataset = prepare_dataset(config, use_mmpr_dataset)
    
    # 数据预处理
    if use_mmpr_dataset:
        tokenized_dataset = dataset.map(
            lambda examples: prepare_mmpr_dataset(
                examples, tokenizer, image_processor, config.max_seq_length
            ),
            batched=True,
            remove_columns=dataset["train"].column_names
        )
    else:
        tokenized_dataset = dataset.map(
            lambda examples: prepare_reward_training_dataset(
                examples, tokenizer, image_processor, config.max_seq_length
            ),
            batched=True,
            remove_columns=dataset["train"].column_names
        )
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps" if "validation" in dataset else "no",
        eval_steps=config.eval_steps if "validation" in dataset else None,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16,
        bf16=config.bf16,
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=config.remove_unused_columns,
        report_to="wandb" if config.use_wandb else "none",
    )
    
    # 设置wandb
    if config.use_wandb:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name or f"{config.model_name_or_path.split('/')[-1]}-reward",
            entity=config.wandb_entity,
            config=asdict(config),
            # 添加更多标签和元数据
            tags=["reward-model", "mmpr", "qwen2-vl" if "qwen" in config.model_name_or_path.lower() else "other-model"],
            notes="训练多模态奖励模型，使用MMPR数据集" if use_mmpr_dataset else "训练多模态奖励模型",
        )
        
        # 记录模型架构
        if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
            wandb.config.update({"model_config": model.base_model.config.to_dict()}, allow_val_change=True)
        
        # 记录数据集信息
        wandb.config.update({
            "dataset_size": len(dataset["train"]),
            "validation_size": len(dataset["validation"]) if "validation" in dataset else 0,
            "use_mmpr_dataset": use_mmpr_dataset,
        }, allow_val_change=True)
        
        # 如果使用LoRA，记录LoRA配置
        if use_lora:
            wandb.config.update({
                "use_lora": use_lora,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_target_modules": target_modules,
            }, allow_val_change=True)
            
        # 记录一个样本示例
        if use_mmpr_dataset and len(dataset["train"]) > 0:
            sample = dataset["train"][0]
            if "image" in sample and sample["image"] is not None and hasattr(sample["image"], "convert"):
                wandb.log({
                    "sample_image": wandb.Image(sample["image"], 
                                               caption="样本图像示例"),
                    "sample_question": sample.get("question", ""),
                    "sample_chosen": sample.get("chosen", ""),
                    "sample_rejected": sample.get("rejected", ""),
                })
                
        # 监控模型参数和梯度
        if config.wandb_watch != "false":
            wandb.watch(
                model, 
                log=config.wandb_watch,  # "gradients", "parameters", "all"
                log_freq=config.wandb_log_steps,
                log_graph=True,
            )
    
    # 创建Trainer
    if use_mmpr_dataset:
        trainer = PreferenceTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation", None),
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset.get("validation", None),
            compute_metrics=compute_metrics if "validation" in dataset else None,
        )
    
    # 训练模型
    trainer.train()
    
    # 保存模型
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info(f"模型已保存到 {config.output_dir}")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    main() 