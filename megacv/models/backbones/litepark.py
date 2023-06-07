#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-16 11:07:11
"""

import math
from typing import Any, Dict, List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import BACKBONES


class MaxoutBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        norm_cfg: Dict[str, Any] = {"type": "BN"},
        act_cfg: Dict[str, Any] = {"type": "ReLU"},
    ):
        super().__init__()
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        if self.stride == 1:
            self.inner_channels = out_channels
            self.out_channels = self.inner_channels + self.inner_channels // 2
        else:
            self.inner_channels = out_channels // 2
            self.out_channels = out_channels

        self.build_layers()

    def build_layers(self):
        res1_channels = self.inner_channels // 2 if self.stride == 1 else self.in_channels
        self.residual1 = ConvModule(in_channels=res1_channels,
                                    out_channels=self.inner_channels,
                                    kernel_size=(1, 3),
                                    padding=(0, 1),
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)
        self.residual2 = ConvModule(in_channels=self.inner_channels,
                                    out_channels=self.inner_channels,
                                    kernel_size=3,
                                    padding=1,
                                    stride=self.stride,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)
        self.residual3 = ConvModule(in_channels=self.inner_channels,
                                    out_channels=self.inner_channels,
                                    kernel_size=(3, 1),
                                    padding=(1, 0),
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg)
        if self.stride == 2:
            self.proj = ConvModule(in_channels=self.in_channels,
                                   out_channels=self.inner_channels,
                                   kernel_size=3,
                                   padding=1,
                                   stride=self.stride,
                                   norm_cfg=self.norm_cfg,
                                   act_cfg=self.act_cfg)
        else:
            self.shuffle_conv = ConvModule(in_channels=self.in_channels,
                                           out_channels=self.inner_channels,
                                           kernel_size=1,
                                           norm_cfg=None,
                                           act_cfg=None,
                                           bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            inputs = self.shuffle_conv(inputs)
            x_proj, x_main = inputs[:, :self.inner_channels // 2], inputs[:, self.inner_channels // 2:]
        else:
            x_proj, x_main = inputs, inputs

        x_main = self.residual1(x_main)
        x_main = self.residual2(x_main)
        x_main = self.residual3(x_main)
        if self.stride == 2:
            x_proj = self.proj(x_proj)
        return torch.cat([x_proj, x_main], dim=1)


@BACKBONES.register_module()
class LitePark(nn.Module):

    def __init__(
        self,
        blocks: List[int] = [2, 2, 2, 2],
        channels: List[int] = [64, 128, 256, 512],
        norm_cfg: Dict[str, Any] = {"type": "BN"},
        act_cfg: Dict[str, Any] = {"type": "ReLU"},
    ):
        super().__init__()
        self.blocks = blocks
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.build_layers()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def build_layers(self):
        self.layers = nn.ModuleList()
        self.layers.append(ConvModule(in_channels=3,
                                      out_channels=32,
                                      kernel_size=3,
                                      padding=1,
                                      stride=2,
                                      norm_cfg=self.norm_cfg,
                                      act_cfg=self.act_cfg))

        last_channels = 32
        for i, (block, channel) in enumerate(zip(self.blocks, self.channels)):
            stage = nn.Sequential()
            for j in range(block):
                in_channels = last_channels if j == 0 else stage[-1].out_channels
                stride = 2 if j == 0 else 1
                stage.append(MaxoutBlock(in_channels=in_channels,
                                         out_channels=channel,
                                         stride=stride))
            self.layers.append(stage)
            last_channels = stage[-1].out_channels

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        pyramids = []
        for idx, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if idx > 0:
                pyramids.append(inputs)

        return pyramids
