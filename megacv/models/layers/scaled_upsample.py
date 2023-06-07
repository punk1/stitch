#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-07 15:18:40
"""

import math
from typing import Any, Dict

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import LAYERS
from .base_module import BaseModule


@LAYERS.register_module()
class ScaledUpsample(BaseModule):

    """Upsample feature map with conv module, structure like `upsample->conv->upsample->conv`

    Args:
        scale_factor (int): upsample rate, Default: 2
        num_convs (int): number of convs, Default: 1
        in_channels (int): input feature map channels, Default: 32
        out_channels (int): output feature map channels, Default: 32
        kernel_size (int): conv kernel_size, Default: 3
        padding (int): conv padding, Default: 1
        norm_cfg (dict): conv normalize cfg, Default: {"type": "BN"}
        acg_cfg (dict): conv activation cfg, Default: {"type": "ReLU"}
        upsample_cfg (dict): upsample cfg, Default: {"mode": "nearest"}
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 32,
        scale_factor: int = 2,
        num_convs: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
        norm_cfg: Dict[str, Any] = {"type": "BN"},
        act_cfg: Dict[str, Any] = {"type": "ReLU"},
        upsample_cfg: Dict[str, Any] = {"mode": "nearest"},
    ):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(int(math.log(scale_factor, 2)) - 1):
            layer = nn.Sequential()
            layer.append(nn.Upsample(scale_factor=2, **upsample_cfg))
            for j in range(num_convs):
                channels = in_channels if i == 0 and j == 0 else out_channels
                layer.append(ConvModule(in_channels=channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg))
            self.layers.append(layer)
        self.layers.append(nn.Upsample(scale_factor=2, **upsample_cfg))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)
