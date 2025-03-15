#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.reward_model import RewardModel

class ContrastiveRewardModel(RewardModel):
    """使用对比学习的奖励模型"""
    def __init__(self, config, base_model, temperature=0.5, contrastive_weight=0.5):
        super().__init__(config, base_model)
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
    
    def compute_preference_loss(self, chosen_scores, rejected_scores):
        """计算偏好损失和对比损失的组合"""
        # 标准偏好损失
        probs = torch.sigmoid(chosen_scores - rejected_scores)
        preference_loss = -torch.log(probs).mean()
        
        # 对比损失
        batch_size = chosen_scores.size(0)
        if batch_size > 1:  # 只有当批次大小大于1时才计算对比损失
            # 归一化分数
            chosen_norm = chosen_scores / self.temperature
            rejected_norm = rejected_scores / self.temperature
            
            # 计算相似度矩阵
            all_scores = torch.cat([chosen_norm, rejected_norm], dim=0)
            sim_matrix = torch.exp(all_scores.unsqueeze(0) - all_scores.unsqueeze(1))
            
            # 创建标签矩阵：1表示同类（都是chosen或都是rejected），0表示不同类
            labels = torch.cat([
                torch.ones(batch_size, device=chosen_scores.device),
                torch.zeros(batch_size, device=chosen_scores.device)
            ])
            mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            
            # 对角线上的元素是自身的相似度，应该排除
            mask = mask.fill_diagonal_(0)
            
            # 计算对比损失
            pos_sim = sim_matrix * mask
            neg_sim = sim_matrix * (1 - mask)
            
            # 避免除以零
            pos_sum = pos_sim.sum(dim=1)
            neg_sum = neg_sim.sum(dim=1) + 1e-8
            
            # 计算每个样本的对比损失
            contrastive_loss = -torch.log(pos_sum / (pos_sum + neg_sum)).mean()
            
            # 组合两种损失
            return preference_loss + self.contrastive_weight * contrastive_loss
        else:
            # 批次大小为1时，只返回偏好损失
            return preference_loss 