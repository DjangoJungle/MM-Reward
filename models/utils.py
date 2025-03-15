import os
import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional

def set_seed(seed: int):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_model_kwargs(model_name: str) -> Dict:
    """根据模型名称准备相应的参数"""
    kwargs = {}
    
    if "qwen" in model_name.lower():
        # Qwen2-VL特定参数
        kwargs.update({
            "model_type": "qwen2",
        })
    elif "internvl" in model_name.lower():
        # InternVL2.5特定参数
        kwargs.update({
            "model_type": "internvl",
        })
    
    return kwargs

def get_reward_model(model_name_or_path: str, model_type: str = "base", **kwargs) -> torch.nn.Module:
    """获取奖励模型
    
    Args:
        model_name_or_path: 模型名称或路径
        model_type: 奖励模型类型，可选值：
            - "base": 基础奖励模型
            - "enhanced": 增强型奖励模型
            - "pooling": 特征池化奖励模型
            - "contrastive": 对比学习奖励模型
            - "fusion": 多层融合奖励模型
            - "adaptive": 自适应难度加权奖励模型
        **kwargs: 其他参数
    
    Returns:
        torch.nn.Module: 奖励模型实例
    """
    from models.reward_model import RewardModel
    from transformers import Qwen2VLModel, AutoModelForCausalLM
    
    # 根据模型类型导入相应的模型类
    if model_type == "enhanced":
        from models.enhanced_reward_model import EnhancedRewardModel as ModelClass
    elif model_type == "pooling":
        from models.pooling_reward_model import PoolingRewardModel as ModelClass
    elif model_type == "contrastive":
        from models.contrastive_reward_model import ContrastiveRewardModel as ModelClass
    elif model_type == "fusion":
        from models.fusion_reward_model import FusionRewardModel as ModelClass
    elif model_type == "adaptive":
        from models.adaptive_reward_model import AdaptiveRewardModel as ModelClass
    else:
        # 默认使用基础奖励模型
        ModelClass = RewardModel
    
    model_kwargs = prepare_model_kwargs(model_name_or_path)
    model_kwargs.update(kwargs)
    
    # 加载基础模型
    if model_kwargs.get("model_type") == "qwen2":
        base_model = Qwen2VLModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=True,
            **kwargs
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=True,
            **kwargs
        )
    
    # 创建奖励模型
    return ModelClass(base_model.config, base_model, **kwargs) 