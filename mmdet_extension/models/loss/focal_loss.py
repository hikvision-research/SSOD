# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/facebookresearch/unbiased-teacher
"""
CE version of Focal Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


@LOSSES.register_module()
class CEFocalLoss(nn.Module):
    def __init__(self, use_sigmoid=False, gamma=2.0, alpha=0.25, reduction='mean',
                 class_weight=None, loss_weight=1.0):
        super().__init__()
        assert use_sigmoid is False, 'Only ce focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss = F.cross_entropy(cls_score, label, weight=class_weight, reduction='none')
        p = torch.exp(-loss)
        loss = (1 - p) ** self.gamma * loss
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss
