#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-10 17:31:27
"""

import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss

from ..builder import LOSSES


@weighted_loss
def cross_iou_loss(pred, target):
    """CrossIOULoss from LSNet: https://arxiv.org/abs/2104.04899

    Args:
        pred: N * max_objects * 16 (4 points cross-coordinate system)
        target: N * max_objects * 16
    """
    num_points = pred.shape[-1] // 4
    pred = pred.reshape(*pred.shape[:-1], num_points, 4)
    target = target.reshape(*target.shape[:-1], num_points, 4)
    total = torch.stack([pred, target], -1)
    lmax = torch.max(total, dim=-1)[0]
    lmin = torch.min(total, dim=-1)[0]
    overlaps = lmin.sum(dim=-1) / (lmax.sum(dim=-1) + 1e-7)
    return 1 - overlaps.mean(dim=-1)


@LOSSES.register_module()
class CrossIOULoss(nn.Module):

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
        loss = cross_iou_loss(
            pred, target,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
        )
        return self.loss_weight * loss
