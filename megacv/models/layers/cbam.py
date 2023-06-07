#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-07-20 19:39:04
"""

import torch
import torch.nn as nn

from ..builder import LAYERS


class ChannelAttention(nn.Module):

    def __init__(self, in_channels, ratio=16, feat_size=64):
        super().__init__()
        # h, w = feat_size, feat_size if isinstance(feat_size, int) else feat_size
        # kernel_size = stride = h, w
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


@LAYERS.register_module()
class CBAM(nn.Module):

    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x
