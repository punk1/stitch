#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2023-04-14 10:32:22
"""

from typing import Any, Dict

import torch

from ..models.builder import MODELS
from ..utils import CheckpointManager, load_cfg
from .builder import TRAINER
from .trainer import Trainer


@TRAINER.register_module()
class DistillTrainer(Trainer):

    def build_model(self):
        super().build_model()
        cfg = load_cfg(self.cfg.teacher)
        cfg.task_configs = {k: load_cfg(v) for k, v in cfg.task_configs.items()}
        self.teacher_model = MODELS.build(cfg.model)
        self.teacher_model = self.teacher_model.cuda(device=self.current_device).eval()
        ckpt_manager = CheckpointManager(
            model=self.teacher_model,
            resume=cfg.resume,
            pretrained=cfg.pretrained,
            to_cuda=True,
        )
        ckpt_manager.load_model()

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        preds = self.model(inputs)
        with torch.no_grad():
            teacher_preds = self.teacher_model(inputs)
        loss = self.model.compute_loss(preds, inputs)
        kdloss = self.model.compute_kdloss(preds, teacher_preds, inputs)
        loss.update(kdloss)
        return loss
