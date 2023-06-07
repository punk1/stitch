#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-05-12 15:26:09
"""

import logging
from functools import cached_property
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn

from ...utils import load_cfg
from ..builder import BACKBONES, HEADS, MODELS, NECKS, NORMALIZERS

logger = logging.getLogger()


@MODELS.register_module()
class MultiDetector(nn.Module):

    """MultiDetector for multi-task with shared backbone

    Args:
        tasks (dict): config name for sub task
        backbone (dict): backbone cfg
        normalizer (dict): normalizer cfg
        neck (dict): neck cfg
        **kwargs (dict): default detector cfg
    """

    def __init__(
        self,
        backbone: Dict[str, Any],
        normalizer: Dict[str, Any] = None,
        tasks: Dict[str, str] = None,
        **kwargs
    ):
        super().__init__()
        self.backbone = BACKBONES.build(backbone)
        if normalizer is not None:
            self.normalizer = NORMALIZERS.build(normalizer)

        self.necks = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        for key, config in tasks.items():
            cfg = load_cfg(config)
            if cfg.model.neck is not None:
                self.necks[key] = NECKS.build(cfg.model.neck)
            self.heads[key] = HEADS.build(cfg.model.head)

    @cached_property
    def with_normalizer(self):
        return hasattr(self, "normalizer")

    def init_weights(self):
        if hasattr(self.backbone, 'init_weights'):
            logger.warning(f'init weights: {self.backbone.__class__.__name__}')
            self.backbone.init_weights()
        for neck in self.necks:
            if hasattr(neck, 'init_weights'):
                logger.warning(f'init weights: {neck.__class__.__name__}')
                neck.init_weights()
        for head in self.heads:
            if hasattr(head, 'init_weights'):
                logger.warning(f'init weights: {head.__class__.__name__}')
                head.init_weights()

    def compute_loss(self, preds: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.heads[inputs["task"]].compute_loss(preds, inputs)

    def compute_kdloss(
        self,
        preds: Dict[str, Any],
        teacher_preds: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.heads[inputs["task"]].compute_kdloss(preds, teacher_preds)

    def post_process(self, preds: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.heads[inputs["task"]].post_process(preds, inputs)

    def onnx_export(self, feats: Dict[str, Any]) -> List[torch.Tensor]:
        outputs = []
        for task in self.heads:
            tmp = self.necks[task](feats) if task in self.necks else feats
            preds = self.heads[task](tmp)
            res = self.heads[task].onnx_export(preds)
            if isinstance(res, (list, tuple)):
                outputs.extend(res)
            else:
                outputs.append(res)
        return outputs

    def quant_infer(self, feats: Dict[str, Any]) -> List[torch.Tensor]:
        outputs = []
        for task in self.heads:
            tmp = self.necks[task](feats) if task in self.necks else feats
            preds = self.heads[task](tmp)
            for v in preds.values():
                if isinstance(v, (list, tuple)):
                    outputs.extend(v)
                else:
                    outputs.append(v)
        return outputs

    def forward(self, inputs: Dict[str, Any]) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        if self.with_normalizer and torch.onnx.is_in_onnx_export():
            inputs = self.normalizer(inputs)

        if isinstance(inputs, dict):
            img, task = inputs["img"], inputs.get("task")
        else:
            img, task = inputs, None

        feats = self.backbone(img)
        if torch.onnx.is_in_onnx_export():
            return self.onnx_export(feats)

        if task is None:
            return self.quant_infer(feats)

        tmp = self.necks[task](feats) if task in self.necks else feats
        return self.heads[task](tmp)
