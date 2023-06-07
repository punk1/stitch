#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-16 13:34:32
"""

import random
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from megacv.data.builder import TRANSFORMS
from megacv.data.transforms.base_transform import BaseTransform, random_apply


@TRANSFORMS.register_module()
class PLDResize(BaseTransform):
    """Resize image.

    Args:
        size (Union[int, tuple]): final size
        interpolation (InterpolationMode): torchvision.transforms.InterpolationMode
    """

    def __init__(
        self,
        size: Union[int, List, Tuple],
        interpolation: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
    ):
        self.interpolation = interpolation
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        h, w = data['img'].shape[1:]
        scale_factor = self.size[0] / h, self.size[1] / w
        data["scale_factor"] = torch.from_numpy(np.array(scale_factor, dtype=np.float32))
        data["img"] = TF.resize(data["img"], self.size, self.interpolation)
        data["kpts"][..., 0] *= scale_factor[1]
        data["kpts"][..., 1] *= scale_factor[0]
        return data


@TRANSFORMS.register_module()
class PLDRandomResizedCrop(T.RandomResizedCrop):

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        i, j, h, w = self.get_params(data['img'], self.scale, self.ratio)
        data["kpts"][..., 0] -= j
        data["kpts"][..., 1] -= i
        data["kpts"][..., 0] *= self.size[1] / w
        data["kpts"][..., 1] *= self.size[0] / h
        height, width = data['img'].shape[1:]
        scale_factor = self.size[0] / height, self.size[1] / width
        data["scale_factor"] = torch.from_numpy(np.array(scale_factor, dtype=np.float32))
        data['img'] = TF.resized_crop(data['img'], i, j, h, w, self.size, self.interpolation)
        return data


@TRANSFORMS.register_module()
class PLDRandomFlip(BaseTransform):
    """Flip image.

    Args:
        p: The flipping probability
        direction (list): The flipping direction, options: 'horizontal', 'vertical'
    """

    def __init__(
        self,
        p: float = 0.5,
        direction: List[str] = ['horizontal', 'vertical'],
        resort: bool = True,
    ):
        self.p = p
        self.direction = direction
        self.resort = resort

    @random_apply
    def horizontal_flip(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data['img'] = TF.hflip(data['img'])
        w = data['img'].shape[2]
        kpts = data['kpts']
        kpts[..., 0] = w - 1 - kpts[..., 0]
        if self.resort:
            order = [1, 0, 3, 2]
            data['kpts'] = kpts[:, order, :]
        else:
            data['kpts'] = kpts
        return data

    @random_apply
    def vertical_flip(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data['img'] = TF.vflip(data['img'])
        h = data['img'].shape[1]
        kpts = data['kpts']
        kpts[..., 1] = h - 1 - kpts[..., 1]
        if self.resort:
            order = [1, 0, 3, 2]
            data['kpts'] = kpts[:, order, :]
        else:
            data['kpts'] = kpts
        return data

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if 'horizontal' in self.direction:
            data = self.horizontal_flip(data)
        if 'vertical' in self.direction:
            data = self.vertical_flip(data)
        return data


@TRANSFORMS.register_module()
class PLDRandomCrop(BaseTransform):

    def __init__(self, p: float = 0.5, min_crop_ratio: float = 0.5):
        self.p = p
        self.min_crop_ratio = min_crop_ratio

    @random_apply
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img = data['img']
        kpts = data['kpts']
        c, h, w = img.shape
        new_w = random.uniform(self.min_crop_ratio * w, w)
        new_h = random.uniform(self.min_crop_ratio * h, h)

        if kpts.shape[0] == 0:
            left = random.uniform(0, w - new_w)
            top = random.uniform(0, h - new_h)
            patch = np.array([int(left), int(top), int(left + new_w), int(top + new_h)])
        else:
            # ensure at least one center in the image
            centers = kpts.mean(dim=1)
            valid_flags = (centers[:, 0] > 0) & (centers[:, 0] < w) & (centers[:, 1] > 0) & (centers[:, 1] < h)
            valid_kpts = kpts[valid_flags, ...]
            valid_centers = centers[valid_flags, ...]
            if valid_kpts.shape[0] == 0:
                left = random.uniform(0, w - new_w)
                top = random.uniform(0, h - new_h)
                patch = np.array([int(left), int(top), int(left + new_w), int(top + new_h)])
            else:
                (c_left, c_top, _), _ = valid_centers.min(dim=0)
                (c_right, c_bottom, _), _ = valid_centers.max(dim=0)
                boundary_left, boundary_right = max(c_left - new_w, 0), min(c_right, w - new_w)
                boundary_top, boundary_bottom = max(c_top - new_h, 0), min(c_bottom, h - new_h)
                left = random.uniform(boundary_left, boundary_right)
                top = random.uniform(boundary_top, boundary_bottom)
                patch = np.array([int(left), int(top), int(left + new_w), int(top + new_h)])

        data['img'] = data['img'][:, patch[1]:patch[3], patch[0]:patch[2]]
        kpts[..., 0] -= patch[0]
        kpts[..., 1] -= patch[1]
        data['kpts'] = kpts
        return data


@TRANSFORMS.register_module()
class PLDRandomRotate(BaseTransform):

    def __init__(self, p=0.5, angle=30, mean=(0, 0, 0), expand=False):
        self.p = p
        self.angle = angle
        self.mean = mean
        self.expand = expand

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
            expand=self.expand,
        )
        kpts = np.array(data['kpts'])
        if kpts.shape[0] != 0:
            rotate_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
            vis = kpts[:, :, 2:3]
            src_kpts = np.concatenate([kpts[:, :, :2],
                                       np.ones((kpts.shape[0], kpts.shape[1], 1))], axis=-1)
            dst_kpts = np.matmul(rotate_matrix, src_kpts.transpose(0, 2, 1)).transpose(0, 2, 1)
            dst_kpts = np.concatenate([dst_kpts, vis], axis=-1)
            data['kpts'] = torch.from_numpy(dst_kpts)

        return data


@TRANSFORMS.register_module()
class PLDCenterCrop(BaseTransform):

    def __init__(self, size: Union[int, List, Tuple]):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img = TF.center_crop(data["img"], self.size)
        lift_x = (data["img"].shape[2] - self.size[1]) / 2
        lift_y = (data["img"].shape[1] - self.size[1]) / 2
        data["kpts"][..., 0] -= lift_x
        data["kpts"][..., 1] -= lift_y
        data["img"] = img
        return data
