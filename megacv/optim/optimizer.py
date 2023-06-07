#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-06-14 11:32:26
"""

import inspect

import torch
from timm.optim.adan import Adan
from timm.optim.lion import Lion

from .builder import OPTIMIZER

OPTIMIZER.register_module()(Adan)
OPTIMIZER.register_module()(Lion)


def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            OPTIMIZER.register_module()(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


register_torch_optimizers()
