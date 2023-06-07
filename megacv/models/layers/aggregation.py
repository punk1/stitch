#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-08 10:48:03
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import LAYERS
from .base_module import BaseModule


@LAYERS.register_module()
class Aggregation(BaseModule):

    """Aggregation multiple feature map

    Args:
        in_channels (list[int]): input feature map channels, Default: 32
        out_channels (int): output feature map channels, Default: 32
        num_convs (int): number of convs, Default: 1
        kernel_size (int): conv kernel_size, Default: 3
        padding (int): conv padding, Default: 1
        norm_cfg (dict): conv normalize cfg, Default: {"type": "BN"}
        acg_cfg (dict): conv activation cfg, Default: {"type": "ReLU"}
        upsample_cfg (dict): upsample cfg, Default: {"mode": "nearest"}
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 32,
        num_convs: int = 1,
        kernel_size: int = 3,
        padding: int = 1,
        norm_cfg: Dict[str, Any] = {"type": "BN"},
        act_cfg: Dict[str, Any] = {"type": "ReLU"},
        upsample_cfg: Dict[str, Any] = {"mode": "nearest"},
    ):
        super().__init__()
        self.upsample_cfg = upsample_cfg.copy()
        self.in_channels = in_channels
        self.layers = nn.ModuleList()
        for i in range(len(in_channels)):
            self.layers.append(ConvModule(in_channels=in_channels[i],
                                          out_channels=out_channels if i == 0 else in_channels[i - 1],
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          norm_cfg=norm_cfg,
                                          act_cfg=act_cfg))
        # self.conv = ConvModule(in_channels=out_channels * len(in_channels),
        #                        out_channels=out_channels,
        #                        kernel_size=kernel_size,
        #                        padding=padding,
        #                        norm_cfg=norm_cfg,
        #                        act_cfg=act_cfg)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        inners = [inputs[-1]]
        for i in range(len(self.in_channels) - 1, 0, -1):
            prev_shape = inputs[i - 1].shape[2:]
            inner_out = self.layers[i](F.interpolate(inners[-1], size=prev_shape, **self.upsample_cfg))
            inners.append(inputs[i - 1] + inner_out)

        return self.layers[0](inners[-1])

        # feats = [F.interpolate(x, size=inputs[0].shape[2:], **self.upsample_cfg) for x in inputs]
        # feats = [layer(x) for layer, x in zip(self.layers, feats)]
        # feats = torch.cat(feats, dim=1)
        # return self.conv(feats)
