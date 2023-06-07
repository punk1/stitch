#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BaseDetector

Author:     kaizhang
Created at: 2022-04-01 11:28:39
"""

import logging
from functools import cached_property
from typing import Any, Dict

import torch
import torch.nn as nn

from ..builder import BACKBONES, HEADS, MODELS, NECKS, NORMALIZERS

logger = logging.getLogger()


@MODELS.register_module()
class BaseDetector(nn.Module):

    """An simple detector with model structure `backbone->neck->head`.

    Args:
        backbone (dict): backbone cfg
        normalizer (dict): normalizer cfg
        neck (dict): neck cfg
        head (dict): head cfg
        **kwargs (dict): default detector cfg
    """

    def __init__(
        self,
        backbone: Dict[str, Any],
        normalizer: Dict[str, Any] = None,
        neck: Dict[str, Any] = None,
        head: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__()
        self.backbone = BACKBONES.build(backbone)
        if normalizer is not None:
            self.normalizer = NORMALIZERS.build(normalizer)
        if neck is not None:
            self.neck = NECKS.build(neck)

        assert head is not None, "Except type(head)=dict, got None"
        self.head = HEADS.build(head)
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.init_weights()

    @cached_property
    def with_normalizer(self):
        return hasattr(self, "normalizer")

    @cached_property
    def with_neck(self):
        return hasattr(self, "neck")

    def init_weights(self):
        if hasattr(self.backbone, 'init_weights'):
            logger.warning(f'init weights: {self.backbone.__class__.__name__}')
            self.backbone.init_weights()
        if self.with_neck and hasattr(self.neck, 'init_weights'):
            logger.warning(f'init weights: {self.neck.__class__.__name__}')
            self.neck.init_weights()
        if hasattr(self.head, 'init_weights'):
            logger.warning(f'init weights: {self.head.__class__.__name__}')
            self.head.init_weights()

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.with_normalizer and torch.onnx.is_in_onnx_export():
            inputs = self.normalizer(inputs)

        feats = self.backbone(inputs["img"])
        if self.with_neck:
            feats = self.neck(feats)

        preds = self.head(feats)
        if torch.onnx.is_in_onnx_export():
            return self.head.onnx_export(preds)

        if self.training:
            return self.head.compute_loss(preds, inputs)
        else:
            return self.head.post_process(preds, inputs)
