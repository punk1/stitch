#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-06 18:58:23
"""

import math

import torch
import torch.optim as optim

from .builder import LR_SCHEDULER


def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.
    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = math.cos(math.pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out


def do_warmup(lrs, step, warmup_method, warmup_step, warmup_factor):
    if warmup_method == 'linear_':
        alpha = step / warmup_step if step < warmup_step else 1.
        return [(warmup_factor * (1. - alpha) + alpha) * base_lr
                for base_lr in lrs]
    elif warmup_method == 'constant':
        return [warmup_factor * base_lr for base_lr in lrs]
    elif warmup_method == 'linear':
        k = (1. - min(1., step / warmup_step)) * (1. - warmup_factor)
        return [(1. - k) * base_lr for base_lr in lrs]
    elif warmup_method == 'exp':
        k = warmup_factor**(1. - min(1., step / warmup_step))
        return [k * base_lr for base_lr in lrs]
    else:
        raise ValueError(f'{warmup_method} is not supported')
        return lrs


def warmup(cls):

    class newCls(cls):

        def __init__(self,
                     *args,
                     warmup_method='linear',
                     warmup_factor=0.,
                     warmup_epoch=0.,
                     min_lr=1e-12,
                     steps_per_epoch=None,
                     **kwargs):
            assert steps_per_epoch is not None, 'steps_per_epoch must be defined'
            self.warmup_method = warmup_method
            self.warmup_factor = warmup_factor
            self.warmup_epoch = warmup_epoch
            self.min_lr = min_lr
            self.steps_per_epoch = steps_per_epoch
            self.warmup_step = int(warmup_epoch * steps_per_epoch)
            super().__init__(*args, **kwargs)

        def get_lr(self):
            if self.last_epoch < self.warmup_step:
                lrs = do_warmup(self.base_lrs, self.last_epoch, self.warmup_method, self.warmup_step, self.warmup_factor)
            else:
                lrs = super()._get_closed_form_lr()
            return [max(lr, self.min_lr) for lr in lrs]

    return newCls


@LR_SCHEDULER.register_module('StepLR')
@warmup
class StepLR(optim.lr_scheduler.StepLR):

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        step_size = int(self.steps_per_epoch * step_size)
        super().__init__(optimizer, step_size, gamma, last_epoch)


@LR_SCHEDULER.register_module('MultiStepLR')
@warmup
class MultiStepLR(optim.lr_scheduler.MultiStepLR):

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        milestones = [int(self.steps_per_epoch * x) for x in milestones]
        super().__init__(optimizer, milestones, gamma, last_epoch)


@LR_SCHEDULER.register_module('ExponentialLR')
@warmup
class ExponentialLR(optim.lr_scheduler.ExponentialLR):

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch - self.warmup_step)
                for base_lr in self.base_lrs]


@LR_SCHEDULER.register_module('CosineAnnealingLR')
@warmup
class CosineAnnealingLR(optim.lr_scheduler.CosineAnnealingLR):

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        super().__init__(optimizer, T_max, eta_min, last_epoch)
        self.T_max = int(self.steps_per_epoch * T_max)

    def _get_closed_form_lr(self):
        return [self.eta_min + ((base_lr - self.eta_min) if self.last_epoch < self.T_max else 0)
                * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_step) / (self.T_max - self.warmup_step))) / 2
                for base_lr in self.base_lrs]


@LR_SCHEDULER.register_module('PolyLR')
@warmup
class PolyLR(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, T_max, power=0.3, last_epoch=-1):
        self.T_max = int(T_max * self.steps_per_epoch)
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        return [base_lr * max(0, 1 - (self.last_epoch - self.warmup_step) / self.T_max)
                ** self.power for base_lr in self.base_lrs]


@LR_SCHEDULER.register_module('CyclicCosineLR')
@warmup
class CyclicCosineLR(optim.lr_scheduler._LRScheduler):

    def __init__(
            self,
            optimizer,
            T_max,
            cyclic_times,
            target_ratio=(0.1, 1),
            step_ratio_up=1.0,
            last_epoch=-1):
        self.target_ratio = target_ratio
        self.cyclic_times = cyclic_times
        self.step_ratio_up = step_ratio_up
        self.lr_phases = []

        max_iters = int(T_max * self.steps_per_epoch)
        max_iter_per_phase = max_iters // self.cyclic_times
        iter_up_phase = int(self.step_ratio_up * max_iter_per_phase)
        self.lr_phases.append(
            [0, iter_up_phase, max_iter_per_phase, 1, self.target_ratio[0]])
        self.lr_phases.append([
            iter_up_phase, max_iter_per_phase, max_iter_per_phase,
            self.target_ratio[0], self.target_ratio[1]
        ])
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        curr_iter = self.last_epoch
        for (start_iter, end_iter, max_iter_per_phase, start_ratio, end_ratio) in self.lr_phases:
            curr_iter %= max_iter_per_phase
            if start_iter <= curr_iter < end_iter:
                progress = curr_iter - start_iter
                return [annealing_cos(base_lr * start_ratio,
                                      base_lr * end_ratio,
                                      progress / (end_iter - start_iter)) for base_lr in self.base_lrs]


@LR_SCHEDULER.register_module('ReduceLROnPlateau')
@warmup
class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):

    def _get_closed_form_lr(self):
        return self._last_lr

    def get_last_lr(self):
        return self._last_lr
