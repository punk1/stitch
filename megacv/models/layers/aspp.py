#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2023-05-08 19:08:16
"""

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from .base_module import BaseModule


class ASPP(BaseModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.
    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super().__init__()
        self.layers = nn.ModuleList()
        for dilation in dilations:
            self.layers.append(
                ConvModule(
                    in_channels,
                    channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.bottleneck = ConvModule(
            len(dilations) * channels,
            channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self.layers:
            aspp_outs.append(aspp_module(x))

        aspp_outs = torch.cat(aspp_outs, dim=1)
        return self.bottleneck(aspp_outs)
