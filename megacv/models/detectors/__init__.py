#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Detector interface.

Author:     kaizhang
Created at: 2022-04-01 11:28:39
"""

from .base_detector import BaseDetector
from .multi_detector import MultiDetector

__all__ = [
    "BaseDetector",
    "MultiDetector",
]
