#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.reward_model import RewardModel

class AdaptiveRewardModel(RewardModel):
    """自适应难度加权的奖励模型"""
    def __init__(self, config, base_model, alpha=1.0, beta=0.5, gamma=2.0):
        super().__init__(config, base_model)
        
        # 自适应难度加权参数
        self.alpha = alpha  # 基础损失权重
        self.beta = beta    # 难度加权系数
        self.gamma = gamma  # 聚焦参数
        
        # 使用更复杂的评分头
        self.score_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.score_head.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def compute_preference_loss(self, chosen_scores, rejected_scores):
        """计算带有自适应难度加权的偏好损失"""
        # 计算标准偏好损失
        score_diff = chosen_scores - rejected_scores
        probs = torch.sigmoid(score_diff)
        standard_loss = -torch.log(probs)
        
        # 计算样本难度权重
        # 难度由分数差异的绝对值决定：差异小的样本更难区分
        with torch.no_grad():
            difficulty = torch.exp(-self.beta * torch.abs(score_diff))
            
            # 使用Focal Loss的思想，根据预测概率调整权重
            # 对于预测错误的样本（概率接近0）给予更高的权重
            pt = probs.clone().detach()
            focal_weight = (1 - pt) ** self.gamma
            
            # 组合难度权重和focal权重
            adaptive_weight = difficulty * focal_weight
            
            # 归一化权重
            adaptive_weight = adaptive_weight / (adaptive_weight.sum() + 1e-8) * len(adaptive_weight)
        
        # 应用自适应权重
        weighted_loss = (adaptive_weight * standard_loss).mean()
        
        return self.alpha * weighted_loss
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """前向传播"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 获取最后一个非填充token的隐藏状态
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(outputs.last_hidden_state.size(0), device=outputs.last_hidden_state.device)
        last_hidden_states = outputs.last_hidden_state[batch_indices, last_token_indices]
        
        # 计算奖励分数
        scores = self.score_head(last_hidden_states).squeeze(-1)
        
        return scores 