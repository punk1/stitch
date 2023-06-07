#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-11 10:49:41
"""

from mmengine.registry import Registry

LR_SCHEDULER = Registry("megacv.lr_scheduler")
OPTIMIZER = Registry("megacv.optimizer")
