#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-07-21 08:46:05
"""

import math
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import LAYERS
from .base_module import BaseModule


@LAYERS.register_module()
class PPM(BaseModule):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feat_size: Tuple[int],
        ratio: int = 16,
        sizes: List[int] = [1, 2, 3, 6],
        upsample_cfg: Dict[str, Any] = {"mode": "nearest"},
        norm_cfg: Dict[str, Any] = {"type": "BN"},
        act_cfg: Dict[str, Any] = {"type": "ReLU"},
    ):
        super().__init__()
        hidden_channels = in_channels // ratio
        h, w = [feat_size, feat_size] if isinstance(feat_size, int) else feat_size
        self.upsample_cfg = upsample_cfg.copy()
        self.stages = nn.ModuleList()
        self.stages.append(ConvModule(in_channels, hidden_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))
        for size in sizes:
            kernel_size = math.ceil(h / size), math.ceil(w / size)
            stride = h // size, w // size
            self.stages.append(nn.Sequential(
                nn.AvgPool2d(kernel_size, stride),
                ConvModule(in_channels, hidden_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            ))
        self.final_conv = ConvModule(
            hidden_channels * (len(sizes) + 1),
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, inputs: torch.Tensor):
        feats = []
        size = inputs.shape[2:]
        feats.append(self.stages[0](inputs))
        for stage in self.stages[1:]:
            feats.append(F.interpolate(stage(inputs), size=size, **self.upsample_cfg))
        return self.final_conv(torch.concat(feats, dim=1))
