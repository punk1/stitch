#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2023-03-23 16:01:04
"""

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import LAYERS
from .base_module import BaseModule


@LAYERS.register_module()
class SPPF(BaseModule):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(
        self,
        in_channels,
        out_channels,
        k=5,
        norm_cfg={"type": "BN"},
        act_cfg={"type": "ReLU"},
    ):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = ConvModule(in_channels, c_, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv2 = ConvModule(c_ * 4, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
