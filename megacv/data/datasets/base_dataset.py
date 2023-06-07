#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BaseDataset.

Author:     kaizhang
Created at: 2022-04-01 13:07:21
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.dataset import T_co

from ..builder import TRANSFORMS


class BaseDataset(Dataset[T_co], ABC):
    """An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Different from torch, we recommend all subclass
    overwrite :meth:`__len__`, which is expected to return the size of the dataset by
    many :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    Args:
        mode (str): Mode in ("train", "val", "test")
        transforms (list): List of transforms
        **kwargs (dict): default data cfg
    """

    VALID_MODE = ("train", "val", "test")

    def __init__(
        self,
        mode: str,
        transforms: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if mode not in self.VALID_MODE:
            raise ValueError(f"Expect mode in {self.VALID_MODE}, but got {mode}.")

        self.logger = logging.getLogger()
        self.mode = mode
        self.transforms = None
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.load_annos()

        if isinstance(transforms, dict) and mode in transforms:
            if isinstance(transforms[mode], dict):
                self.transforms = TRANSFORMS.build(transforms[mode])
            elif isinstance(transforms[mode], (list, tuple)):
                self.transforms = TRANSFORMS.build({"type": "Compose", "transforms": transforms[mode]})
            else:
                raise ValueError(f"Except transforms type `dict` or `list`, but got {type(transforms)}")

    def __add__(self, other: Dataset[T_co]) -> ConcatDataset[T_co]:
        """Concat Datasets."""
        return ConcatDataset([self, other])

    @abstractmethod
    def load_annos(self) -> None:
        """Load annotations."""
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get one data item."""
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset."""
        raise NotImplementedError()
