#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-07-29 18:57:14
"""

from typing import Tuple

import torch.nn as nn
from torch import Tensor


class DropPath(nn.Module):
    def __init__(self, keep_prob: float = 0.5, inplace: bool = False):
        super().__init__()
        self.keep_prob = keep_prob
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.keep_prob > 0:
            mask_shape: Tuple[int] = (x.shape[0],) + (1,) * (x.ndim - 1)
            # remember tuples have the * operator -> (1,) * 3 = (1,1,1)
            mask: Tensor = x.new_empty(mask_shape).bernoulli_(self.keep_prob)
            mask.div_(self.keep_prob)
            if self.inplace:
                x.mul_(mask)
            else:
                x = x * mask
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
