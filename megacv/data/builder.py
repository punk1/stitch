#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data register.

Author:     kaizhang
Created at: 2022-04-01 11:28:39
"""


from mmengine.registry import Registry

TRANSFORMS = Registry("megacv.transforms")
DATASETS = Registry("megacv.datasets")
DATALOADER = Registry("megacv.dataloader")
