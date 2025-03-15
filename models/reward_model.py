#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from transformers import PreTrainedModel, PretrainedConfig

class RewardModel(nn.Module):
    """多模态奖励模型"""
    def __init__(self, config, base_model):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # 创建评分头
        hidden_size = getattr(config, "hidden_size", 768)
        self.score_head = nn.Linear(hidden_size, 1)
        
        # 初始化评分头权重
        torch.nn.init.normal_(self.score_head.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask, pixel_values=None):
        """前向传播函数"""
        # 确保输入数据类型与模型一致
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.base_model.dtype)
        
        # 获取基础模型的输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # 获取最后一层隐藏状态
        last_hidden_states = outputs.hidden_states[-1]
        
        # 获取序列中最后一个token的隐藏状态
        sequence_lengths = torch.sum(attention_mask, dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        
        # 收集每个序列的最后一个token的隐藏状态
        last_token_hidden_states = last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]
        
        # 通过scorehead计算奖励分数
        scores = self.score_head(last_token_hidden_states).squeeze(-1)
        
        return scores
    
    def compute_preference_loss(self, chosen_scores, rejected_scores):
        """计算偏好损失"""
        # 使用交叉熵损失
        probs = torch.sigmoid(chosen_scores - rejected_scores)
        log_probs = torch.log(probs)
        loss = -log_probs.mean()
        
        return loss
    
    def save_pretrained(self, save_directory):
        """保存模型"""
        # 保存基础模型
        self.base_model.save_pretrained(save_directory)
        
        # 保存评分头
        score_head_path = f"{save_directory}/score_head.pt"
        torch.save(self.score_head.state_dict(), score_head_path) 