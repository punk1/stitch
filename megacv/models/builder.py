#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model register.

Author:     kaizhang
Created at: 2022-04-01 11:28:39
"""

import timm
from mmdet.registry import MODELS as MMDET_MODELS
from mmengine.registry import Registry
from mmseg.registry import MODELS as MMSEG_MODELS

MMDET_MODELS._add_child(MMSEG_MODELS)
MMSEG_MODELS.parent = MMDET_MODELS
MODELS = Registry("models", parent=MMSEG_MODELS, scope="megacv")
MODELS.register_module(module=timm.create_model, name='TimmModel')

BACKBONES = MODELS
NECKS = MODELS
LOSSES = MODELS
HEADS = MODELS
LAYERS = MODELS
OPS = MODELS
NORMALIZERS = MODELS
