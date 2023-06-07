#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-05-11 15:11:42
"""

from typing import Any, Dict

import torch
import torch.nn.functional as F

from megacv.models.builder import HEADS
from megacv.models.heads import PLDHead


@HEADS.register_module()
class SegHead(PLDHead):

    def compute_kdloss(self, preds: Dict[str, Any], teacher_preds: Dict[str, Any]) -> Dict[str, Any]:
        losses = {
            "kd_loss": self.losses["kd_seg"](preds["seg"], teacher_preds["seg"])
        }
        return losses

    def compute_loss(self, preds: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        mask = preds['seg']
        target = inputs["seg"].squeeze(1).long()
        if mask.shape[2:] != target.shape[1:]:
            mask = F.interpolate(mask, size=target.shape[1:], **self.upsample_cfg)

        losses = {
            "ce_loss": self.losses["seg"](mask, target),
        }
        return losses

    def post_process(self, preds: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        if preds['seg'].shape[2:] != inputs['img'].shape[2:]:
            preds['seg'] = F.interpolate(preds['seg'], size=inputs['img'].shape[2:], **self.upsample_cfg)

        return {"seg": torch.argmax(preds['seg'], dim=1)}

    def onnx_export(self, preds: Dict[str, Any]) -> torch.Tensor:
        return torch.argmax(preds['seg'], dim=1)
