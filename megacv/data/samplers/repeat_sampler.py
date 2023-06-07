#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-15 11:14:15
"""

from torch.utils.data.sampler import Sampler, T_co


class RepeatSampler:
    """Repeat sampler

    Args:
        sampler (Sampler): default sampler
        start_epoch (int): start epoch, Default: 0
        total_epochs (int): times to repeat, Default: 1
    """

    def __init__(self, sampler: Sampler[T_co], start_epoch: int = 0, total_epochs: int = 1):
        self.sampler = sampler
        self.start_epoch = start_epoch
        self.total_epochs = total_epochs

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        idx = self.start_epoch
        while idx < self.total_epochs:
            if hasattr(self.sampler.sampler, "set_epoch"):
                self.sampler.sampler.set_epoch(idx)
            idx += 1
            yield from iter(self.sampler)
