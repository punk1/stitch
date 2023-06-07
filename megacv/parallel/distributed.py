#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2023-03-01 18:11:45
"""

import torch.nn as nn

from .scatter_gather import scatter_kwargs


class DistributedDataParallel(nn.parallel.DistributedDataParallel):
    """An extension of nn.parallel.DistributedDataParallel.
    Extends state_dict and load_state_dict function.
    """

    def __init__(self, module, device_ids=None, dim=0, **kwargs):
        kwargs.setdefault('bucket_cap_mb', 128)
        super().__init__(module, device_ids=device_ids, dim=dim, **kwargs)
        self.dim = dim

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.module.named_parameters(*args, **kwargs)

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        return getattr(self.module, name)
