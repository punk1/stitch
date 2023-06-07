#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-07-12 18:54:32
"""

import math
from functools import wraps


def enabled(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.enabled:
            return func(self, *args, **kwargs)

    return wrapper


class EMAModel:

    def __init__(self, model, decay=0.9999, factor=5000, step_idx=0, enabled=False):
        self.model = model
        self.decay = decay
        self.enabled = enabled
        self.shadow = {}
        self.backup = {}
        self.step_idx = step_idx
        self.decay = lambda x: decay * (1 - math.exp(-x / factor))
        self.register()

    @enabled
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @enabled
    def step(self):
        self.step_idx += 1
        d = self.decay(self.step_idx)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                data = d * self.shadow[name] + (1 - d) * param.data
                self.shadow[name] = data.clone().detach()

    def state_dict(self):
        if self.enabled:
            state_dict = self.model.state_dict()
            ret = {'model_state_dict': state_dict.copy()}
            state_dict.update(self.shadow)
            ret['ema_state_dict'] = state_dict.copy()
            ret['ema_step_idx'] = self.step_idx
            return ret
        else:
            return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        self.step_idx = state_dict.get('ema_step_idx', 0)
        if self.enabled and 'ema_state_dict' in state_dict:
            return self.model.load_state_dict(state_dict['ema_state_dict'], strict=strict)
        elif 'model_state_dict' in state_dict:
            return self.model.load_state_dict(state_dict['model_state_dict'], strict=strict)
        else:
            return self.model.load_state_dict(state_dict, strict=strict)

    @enabled
    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    @enabled
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
