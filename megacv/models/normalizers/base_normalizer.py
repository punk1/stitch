#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-25 17:40:42
"""

from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import NORMALIZERS


@NORMALIZERS.register_module()
class BaseNormalizer(nn.Module):
    """BaseNormalizer is enabled in onnx export, it's designed to preprocess image in network.
    i.e. resize and normalize the input image.

    Args:
        size (int|list): network image size
        mean (list): Sequence of means for each channel
        std (list): Sequence of standard deviations for each channel
    """

    def __init__(
        self,
        size: Union[int, List[int]],
        mean: List[float] = [123.675, 116.28, 103.53],
        std: List[float] = [58.395, 57.12, 57.375],
    ):
        super().__init__()
        self.size = size
        self.mean = torch.as_tensor(mean).view(1, -1, 1, 1)
        self.std = torch.as_tensor(std).view(1, -1, 1, 1)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # img[..., [2, 1, 0]] is time-consuming
        img = inputs["img"]
        img = img.permute(0, 3, 1, 2)
        img = F.interpolate(img, size=self.size, mode='bilinear')
        # mean = self.mean.repeat(img.shape[0], 1, 1, 1)
        # std = self.std.repeat(img.shape[0], 1, 1, 1)
        img.sub_(self.mean).div_(self.std)
        inputs["img"] = img
        return inputs
