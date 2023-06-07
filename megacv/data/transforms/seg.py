#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-05-11 14:31:25
"""

import random
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from megacv.data.builder import TRANSFORMS
from megacv.data.transforms.base_transform import BaseTransform, random_apply


@TRANSFORMS.register_module()
class SegResize(BaseTransform):

    def __init__(
        self,
        size: Union[int, List, Tuple],
        interpolation: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
    ):
        self.interpolation = interpolation
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        h, w = data['img'].shape[1:]
        scale_factor = self.size[0] / h, self.size[1] / w
        data["scale_factor"] = torch.from_numpy(np.array(scale_factor, dtype=np.float32))
        data["img"] = TF.resize(data["img"], self.size, self.interpolation)
        data["seg"] = TF.resize(data["seg"], self.size, TF.InterpolationMode.NEAREST)
        return data


@TRANSFORMS.register_module()
class SegRandomResizedCrop(T.RandomResizedCrop):

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        i, j, h, w = self.get_params(data['img'], self.scale, self.ratio)
        height, width = data['img'].shape[1:]
        scale_factor = self.size[0] / height, self.size[1] / width
        data['scale_factor'] = torch.from_numpy(np.array(scale_factor, dtype=np.float32))
        data['img'] = TF.resized_crop(data['img'], i, j, h, w, self.size, self.interpolation)
        data['seg'] = TF.resized_crop(data['seg'], i, j, h, w, self.size, TF.InterpolationMode.NEAREST)
        return data


@TRANSFORMS.register_module()
class SegRandomFlip(BaseTransform):

    def __init__(self, p=0.5, direction=['horizontal', 'vertical']):
        self.p = p
        self.direction = direction

    @random_apply
    def horizontal_flip(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data['img'] = TF.hflip(data['img'])
        data['seg'] = TF.hflip(data['seg'])
        return data

    @random_apply
    def vertical_flip(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data['img'] = TF.vflip(data['img'])
        data['seg'] = TF.vflip(data['seg'])
        return data

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if 'horizontal' in self.direction:
            data = self.horizontal_flip(data)
        if 'vertical' in self.direction:
            data = self.vertical_flip(data)
        return data


@TRANSFORMS.register_module()
class SegRandomRotate(BaseTransform):

    def __init__(self, p=0.5, angle=30, mean=(0, 0, 0), ignore_index=255):
        self.p = p
        self.angle = angle
        self.mean = mean
        self.ignore_index = ignore_index

    @random_apply
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        angle = random.uniform(-self.angle, self.angle)
        h, w = data['img'].shape[1:]
        center = ((w - 1) / 2, (h - 1) / 2)
        data['img'] = TF.rotate(
            data['img'],
            -angle,
            interpolation=TF.InterpolationMode.BILINEAR,
            center=center,
            fill=self.mean,
            expand=False,
        )
        data['seg'] = TF.rotate(
            data['seg'],
            -angle,
            interpolation=TF.InterpolationMode.NEAREST,
            center=center,
            fill=[self.ignore_index],
            expand=False,
        )
        return data
