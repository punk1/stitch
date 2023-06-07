#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-08-18 11:54:13
"""

import collections
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import ujson as json

from megacv.data.builder import DATASETS
from megacv.models.postprocessing.pld import kpts2bboxes
from megacv.parallel import DataContainer

from .pld import PLDDataset


@DATASETS.register_module()
class COCODataset(PLDDataset):

    def load_annos(self):
        self.annos = json.load(open(os.path.join(self.ann_root, self.ann_file[self.mode])))
        self.cat_ids = [x['id'] for x in self.annos['categories']]
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.items = self.annos['images']
        annos = collections.defaultdict(list)
        for anno in self.annos['annotations']:
            annos[anno['image_id']].append(anno)
        for item in self.items:
            item['annos'] = annos[item['id']]

    def __len__(self):
        return len(self.items)

    def get_ann_info(self, img_info):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        for ann in img_info['annos']:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        return dict(
            file_name=img_info["file_name"],
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
        )

    def __getitem__(self, index):
        img_info = self.items[index]
        item = self.get_ann_info(img_info)
        img_path = os.path.join(self.ann_root, self.img_root[self.mode], item["file_name"])
        img = cv2.imread(img_path)
        assert img is not None, self.logger.error(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data = {
            "img": TF.to_tensor(img),
            "kpts": torch.from_numpy(item['bboxes']).reshape(-1, 2, 2),
            "labels": torch.from_numpy(item['labels']),
        }

        if self.transforms is not None:
            data = self.transforms(data)

        self.out_shape = [int(self.img_shape[0] * self.out_scale), int(self.img_shape[1] * self.out_scale)]
        if self.out_scale != 1:
            data["kpts"][..., 0] *= self.out_scale
            data["kpts"][..., 1] *= self.out_scale
            data["scale_factor"][0] *= self.out_scale
            data["scale_factor"][1] *= self.out_scale

        bboxes = kpts2bboxes(data["kpts"])
        xxyy = bboxes.reshape(-1, 2, 2).permute(0, 2, 1)
        flag = (xxyy[:, 0] >= 0).any(1) & (xxyy[:, 0] < self.out_shape[1]).any(1) & \
            (xxyy[:, 1] >= 0).any(1) & (xxyy[:, 1] < self.out_shape[0]).any(1)
        data["kpts"] = data["kpts"][flag]
        data["labels"] = data["labels"][flag]

        if self.heatmap:
            data["hm"] = self.gen_heatmap(data["kpts"], data["labels"])
            data["offset"] = self.gen_offset(data["kpts"])
            data["status"] = self.gen_status(data["kpts"], data["labels"])

        data["bboxes"] = DataContainer(bboxes[flag], stack=False)
        data["kpts"] = DataContainer(data["kpts"], stack=False)
        data["labels"] = DataContainer(data["labels"], stack=False)

        return data
