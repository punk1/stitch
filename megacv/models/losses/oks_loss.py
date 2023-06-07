#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-08-25 23:23:24
"""

import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss

from ..builder import LOSSES


@weighted_loss
def oks_loss(kpts_preds, targets, eps=1e-7):
    kpts_targets, bbox_targets = targets
    kpts_preds_x, kpts_targets_x = kpts_preds[:, 0::2], kpts_targets[:, 0::2]
    kpts_preds_y, kpts_targets_y = kpts_preds[:, 1::2], kpts_targets[:, 1::2]
    # kpts_preds_score = kpts_preds[:, 2::3]

    # kpt_mask = (kpts_targets[:, 2::3] != 0)
    # lkptv = F.binary_cross_entropy_with_logits(kpts_preds_score, kpt_mask.float(), reduction='none')
    # OKS based loss
    dis = (kpts_preds_x - kpts_targets_x) ** 2 + (kpts_preds_y - kpts_targets_y) ** 2
    bbox_scale = torch.prod(bbox_targets[:, -2:], dim=1, keepdim=True)
    # kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / torch.sum(kpt_mask != 0)
    oks = torch.exp(-dis / (bbox_scale + eps))
    # lkpt = kpt_loss_factor * ((1 - oks**2) * kpt_mask)
    return 1 - oks**2


@LOSSES.register_module()
class OKSLoss(nn.Module):

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None:
            avg_factor = (weight > 0).sum()
        loss = oks_loss(
            pred, target,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return self.loss_weight * loss
