#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Head interface.

Author:     kaizhang
Created at: 2022-04-01 11:28:39
"""

from .base_head import BaseHead
from .ham import LightHamHead
from .pld import PLDHead
from .seg import SegHead
from .yolox import YOLOXHead

__all__ = [
    "BaseHead",
    "PLDHead",
    "SegHead",
    "YOLOXHead",
    "LightHamHead",
]
