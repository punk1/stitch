#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-13 17:54:08
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

from megacv.evaluators.base_evaluator import BaseEvaluator
from megacv.evaluators.builder import EVALUATORS

from .carspace import CarSpace, CarSpaceEval


@EVALUATORS.register_module()
class PLDEvaluator(BaseEvaluator):

    def update(self, preds: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
        scale_factors = inputs['scale_factor'].cpu().numpy()
        for kpts, scale_factor in zip(preds['kpts'], scale_factors):
            if kpts.shape[0] == 0:
                continue
            for i in range(4):
                kpts[:, 3 * i:3 * i + 1] /= scale_factor[1]
                kpts[:, 3 * i + 1:3 * i + 2] /= scale_factor[0]
        preds["file_name"] = inputs["file_name"]
        return preds

    def evaluate(self, results: Dict[str, Any]) -> Any:
        coco_results = []
        for kpts, file_name in zip(results["kpts"], results["file_name"]):
            for i in range(kpts.shape[0]):
                coco_results.append({
                    "image_id": Path(file_name).stem,
                    "file_name": file_name,
                    "category_id": int(kpts[i, 12]),
                    "score": kpts[i, -1],
                    "keypoints": kpts[i, :12].tolist(),
                })

        self.logger.info(f'coco results: {len(coco_results)}')
        return self.metric_eval(coco_results)

    def metric_eval(self, coco_results: List[Any]):
        if not coco_results:
            return self.logger.error('detect results is None')

        tmp_file = os.path.join(self.save_dir, 'coco_results.json')
        json.dump(coco_results, open(tmp_file, 'w'))

        coco_gt = CarSpace(os.path.join(self.data_cfg.ann_root, self.data_cfg.ann_file.val))
        coco_dt = coco_gt.loadRes(tmp_file)
        coco_eval = CarSpaceEval(coco_gt, coco_dt, self)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize(self.logger)
        res = 'AP@0.9: ' + str(coco_eval.stats[4]) + '; AR@0.9: ' + str(coco_eval.stats[10])
        self.logger.info(f'detection performance: {res}')

        if self.debug:
            badcases = coco_eval.get_badcase()
            self.badcase_display(badcases)

        return coco_eval.stats[4]

    def badcase_display(self, badcases: List[Any]):
        out_dir = os.path.join(self.save_dir, 'badcase')
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        annos = json.load(open(os.path.join(self.data_cfg.ann_root, self.data_cfg.ann_file.val)))
        filepath = {x["id"]: x["file_name"] for x in annos}
        for image_id, badcase in badcases.items():
            savename = os.path.join(out_dir, f'{image_id}.jpg')
            image = Image.open(filepath[image_id])
            draw = ImageDraw.Draw(image)
            self.draw_kpts(draw, badcase['fp'], color='red')
            self.draw_kpts(draw, badcase['fn'], color='yellow')
            image.save(savename)

    @staticmethod
    def draw_kpts(draw, dets, color=None, thickness=3):
        for keypoint, category_id, score in dets:
            font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', 20)
            center_x = (min(keypoint[0::3]) + max(keypoint[0::3])) / 2
            center_y = (min(keypoint[1::3]) + max(keypoint[1::3])) / 2
            draw.text((center_x, center_y), str(round(score, 2)), fill=color, font=font)
            for i in range(4):
                j = (i + 1) % 4
                x1, y1 = keypoint[3 * i: 3 * i + 2]
                x2, y2 = keypoint[3 * j: 3 * j + 2]
                if i == 0:
                    draw.line([(x1, y1), (x2, y2)], width=thickness * 2, fill=color)
                else:
                    draw.line([(x1, y1), (x2, y2)], width=thickness, fill=color)
