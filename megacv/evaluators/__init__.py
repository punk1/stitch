#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluator inferface.

Author:     kaizhang
Created at: 2022-04-01 11:28:22
"""

from .base_evaluator import BaseEvaluator
from .pld.evaluator import PLDEvaluator
from .seg import SegEvaluator

__all__ = [
    "BaseEvaluator",
    "PLDEvaluator",
    "SegEvaluator",
]
