#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BaseTransform.

Author:     kaizhang
Created at: 2022-04-01 11:38:30
"""

import random
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torchvision.transforms as T
from PIL import Image

from ..builder import TRANSFORMS


def random_apply(func):
    """random apply function with probability p."""

    @wraps(func)
    def wrapper(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() <= self.p:
            return func(self, data)
        return data

    return wrapper


class BaseTransform(ABC):
    """Tranform Abstract Interface.

    All subclasses should overwrite :meth:`__call__`
    """

    @abstractmethod
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement of transform.

        Args:
            data (dict): Dataset item to transform
        """
        raise NotImplementedError()


@TRANSFORMS.register_module()
class Compose(BaseTransform):
    """Compose multiple transforms sequentially.

    Args:
        transforms (list): Sequence of transform
            object or config dict to be composed.
    """

    def __init__(
        self,
        transforms: List[Union[Callable, Dict[str, Any]]],
    ):
        """Initialize compose operation."""
        if not isinstance(transforms, list):
            raise ValueError(
                f"Expect type(trainsforms)=list, but got {type(transforms)}."
            )

        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                self.transforms.append(TRANSFORMS.build(transform))
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError("Transform must be callable or Dict.")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            data = transform(data)
        return data


@TRANSFORMS.register_module()
class RandomOrder(Compose):

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            data = self.transforms[i](data)
        return data


@TRANSFORMS.register_module()
class ToTensor(BaseTransform):
    """Convert PIL Image and np.ndarray to torch.Tensor.

    Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, ):
        """Initialize."""
        self.transform = T.ToTensor()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in data.items():
            if isinstance(value, (np.ndarray, Image.Image)):
                data[key] = self.transform(value)
        return data


@TRANSFORMS.register_module()
class ColorJitter(BaseTransform):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        key (str): key in data to apply transform.
        p: probability to apply this transform
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(
        self,
        key: str = "img",
        p: float = 0.5,
        brightness: Union[float, List, Tuple] = 0.5,
        contrast: Union[float, List, Tuple] = 0.5,
        saturation: Union[float, List, Tuple] = 0.5,
        hue: Union[float, List, Tuple] = 0.3,
    ):
        self.key = key
        self.p = p
        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)

    @random_apply
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.key] = self.transform(data[self.key])
        return data


@TRANSFORMS.register_module()
class RandomErasing(BaseTransform):
    """Randomly selects a rectangle region in an torch Tensor image and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
        key (str): key in data to apply transform.
        p: probability that the random erasing operation will be performed.
        scale: range of proportion of erased area against input image.
        ratio: range of aspect ratio of erased area.
        value: erasing value. Default is 0. If a single int, it is used to
           erase all pixels. If a tuple of length 3, it is used to erase
           R, G, B channels respectively.
           If a str of 'random', erasing each pixel with random values.
        inplace: boolean to make this transform inplace. Default set to False.
    """

    def __init__(
        self,
        key: str = "img",
        p: float = 0.5,
        scale: Tuple[float] = (0.01, 0.05),
        ratio: Tuple[float] = (0.3, 3.3),
        value: Union[float, str, Tuple] = 0,
        inplace: bool = False,
    ):
        self.key = key
        self.transform = T.RandomErasing(p, scale, ratio, value, inplace)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.key] = self.transform(data[self.key])
        return data


@TRANSFORMS.register_module()
class Normalize(BaseTransform):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        key (str): key in data to apply transform.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(
        self,
        key: str = "img",
        mean: Union[List, Tuple] = [0.485, 0.456, 0.406],
        std: Union[List, Tuple] = [0.229, 0.224, 0.225],
        inplace: bool = False,
    ):
        self.key = key
        self.transform = T.Normalize(mean, std, inplace)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data[self.key] = self.transform(data[self.key])
        return data
