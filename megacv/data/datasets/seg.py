#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-05-10 17:03:05
"""

import json
import os

import cv2
import torch
import torchvision.transforms.functional as TF

from megacv.data.builder import DATASETS
from megacv.data.datasets import BaseDataset
from megacv.parallel import DataContainer


@DATASETS.register_module()
class SegDataset(BaseDataset):

    def load_annos(self):
        self.items = []
        ann_file = self.ann_file[self.mode]
        if isinstance(ann_file, list):
            for x in ann_file:
                self.items.extend(json.load(open(os.path.join(self.ann_root, x))))
        else:
            self.items.extend(json.load(open(os.path.join(self.ann_root, ann_file))))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        img = cv2.imread(item["file_name"])
        assert img is not None, self.logger.error(item['file_name'])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = cv2.imread(item["mask_name"], cv2.IMREAD_UNCHANGED)

        data = {
            "img": TF.to_tensor(img),
            "seg": torch.from_numpy(seg)[None, ...],
            "mask": torch.from_numpy(seg)[None, ...],
        }
        if self.transforms is not None:
            data = self.transforms(data)

        data["file_name"] = DataContainer(item["file_name"], cpu_only=True)
        return data
