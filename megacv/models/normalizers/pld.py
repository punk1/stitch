#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-25 17:22:58
"""

from typing import Any, Dict

import torch.nn.functional as F

from megacv.models.builder import NORMALIZERS
from megacv.models.normalizers.base_normalizer import BaseNormalizer


@NORMALIZERS.register_module()
class PLDNormalizer(BaseNormalizer):

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        img = inputs["img"]
        img = img.permute(0, 3, 1, 2)
        img[..., 360:660, 436:558] = 0
        img = F.interpolate(img, size=self.size, mode='bilinear')
        img.sub_(self.mean).div_(self.std)
        inputs["img"] = img
        return inputs
