#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-07-28 19:02:28
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import torch.nn.functional as F

from megacv.data.visualizer.pld import draw_seg
from megacv.evaluators.base_evaluator import BaseEvaluator
from megacv.evaluators.builder import EVALUATORS
from megacv.utils.confusion_matrix import ConfusionMatrix


@EVALUATORS.register_module()
class SegEvaluator(BaseEvaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matrix = ConfusionMatrix(self.num_classes)

    def reset(self):
        self.matrix.reset()

    def update(self, preds: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
        size = inputs['mask'].shape[-2:]
        masks = F.interpolate(preds['seg'][:, None].float(), size=size)
        for gt, pred in zip(inputs['mask'], masks):
            self.matrix.update(gt.flatten().long(), pred.flatten().long())
        if self.debug:
            self.vis(preds, inputs)

    def evaluate(self, results: List[Any]) -> Any:
        filename = os.path.join(self.save_dir, 'matrix.pkl')
        pickle.dump(self.matrix.mat.detach().cpu().numpy(), open(filename, 'wb'))
        acc_global, acc, iou = self.matrix.compute()
        miou = iou.mean().item() * 100
        self.logger.info("Acc: {}\nIoU: {}\nmean IoU: {:.1f}".format(
            [f"{i:.1f}" for i in (acc * 100).tolist()],
            [f"{i:.1f}" for i in (iou * 100).tolist()],
            miou,
        ))
        return miou

    def vis(self, preds: Dict[str, Any], inputs: Dict[str, Any]):
        img = inputs['img'].detach().cpu().numpy()
        seg = preds['seg'].detach().cpu().numpy()
        for idx in range(img.shape[0]):
            out_file = Path(self.save_dir) / Path(inputs['file_name'][idx]).name
            draw_seg(img[idx], seg[idx], out_file)
