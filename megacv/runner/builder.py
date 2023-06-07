#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-06 16:28:51
"""

from mmengine.registry import Registry

TRAINER = Registry("megacv.trainer")
INFERER = Registry("megacv.inferer")
QUANTIZER = Registry("megacv.quantizer")
