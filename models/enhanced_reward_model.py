#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.reward_model import RewardModel

class EnhancedRewardModel(RewardModel):
    """增强型评分头的奖励模型"""
    def __init__(self, config, base_model):
        # 不直接调用父类的__init__，而是重新实现
        super(nn.Module, self).__init__()
        self.config = config
        self.base_model = base_model
        
        # 创建增强型评分头
        hidden_size = getattr(config, "hidden_size", 768)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 初始化评分头权重
        for module in self.score_head.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    # forward方法和compute_preference_loss方法与原始RewardModel相同，无需修改 