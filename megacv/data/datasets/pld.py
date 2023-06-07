#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PLD Dataset.

Author:     kaizhang
Created at: 2022-04-01 14:06:05
"""

import collections
import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import ujson as json
from mmdet.models.utils.gaussian_target import gen_gaussian_target

from megacv.data.builder import DATASETS
from megacv.data.datasets import BaseDataset
from megacv.models.postprocessing.pld import kpts2bboxes
from megacv.parallel import DataContainer


@DATASETS.register_module()
class PLDDataset(BaseDataset):

    def load_annos(self):
        images = []
        annotations = []
        ann_file = self.ann_file[self.mode]
        if isinstance(ann_file, list):
            for x in ann_file:
                doc = json.load(open(os.path.join(self.ann_root, x)))
                images.extend(doc['images'])
                annotations.extend(doc['annotations'])
        else:
            doc = json.load(open(os.path.join(self.ann_root, ann_file)))
            images.extend(doc['images'])
            annotations.extend(doc['annotations'])

        res = collections.defaultdict(list)
        for doc in annotations:
            res[doc['image_id']].append(doc)

        self.items = images
        for item in self.items:
            item['kpts'] = [x['keypoints'] for x in res[item['id']]]
            item['labels'] = [x['category_id'] for x in res[item['id']]]

    def __len__(self):
        return len(self.items)

    def gen_heatmap(self, kpts, categories):
        h, w = self.out_shape
        num_points = self.num_points if self.match else 0
        hm_targets = torch.zeros((self.num_classes + num_points, h, w), dtype=torch.float32)
        for kpt, category_id in zip(kpts, categories):
            cx = kpt[..., 0].mean().int()
            cy = kpt[..., 1].mean().int()
            if not (cx >= 0 and cx < w and cy >= 0 and cy < h):
                continue

            for i in range(num_points):
                pt = kpt[i, :2]
                if pt[0] >= 0 and pt[0] < w and pt[1] >= 0 and pt[1] < h:
                    hm_targets[i] = gen_gaussian_target(hm_targets[i], pt.int(), self.radius)

            idx = num_points + category_id
            hm_targets[idx] = gen_gaussian_target(hm_targets[idx], [cx, cy], int(self.radius * 1.5))

        return hm_targets

    def gen_seg(self, kpts, labels):
        h, w = self.out_shape
        seg = np.ones((3 * h, 3 * w), np.uint8) * 255
        kpts = kpts.clone()
        kpts[..., 0] += w
        kpts[..., 1] += h
        for i in range(self.num_classes):
            kpts_i = [kpt for kpt, s in zip(kpts, labels) if s == i]
            if not kpts_i:
                continue
            kpts_i = torch.vstack(kpts_i).reshape(-1, 4, 3)[..., :2].cpu().numpy()
            seg = cv2.fillPoly(seg, kpts_i.astype(np.int32), i)

        seg = seg[h:2 * h, w:2 * w]
        return torch.from_numpy(seg)

    def gen_offset(self, kpts):
        max_objects = self.max_objects
        num_points = self.num_points
        object_num = min(kpts.shape[0], max_objects)
        channel_num = 2 + num_points * 4
        targets = torch.zeros([max_objects, channel_num], dtype=torch.float32)
        if object_num == 0:
            return targets

        h, w = self.out_shape
        centers_int = kpts.mean(dim=1).int()[..., :2]
        offset = kpts[..., :2] - centers_int[:, None, :]
        br_flag = offset >= 0
        tl_flag = offset < 0
        pos_flag = torch.cat([br_flag, tl_flag], dim=-1)
        neg_flag = ~pos_flag
        x_abs = torch.cat([offset.abs(), offset.abs()], dim=-1)
        target = pos_flag * x_abs + neg_flag * x_abs * self.cross_iou_alpha
        target = target[..., [0, 2, 1, 3]]

        x = centers_int[..., 0]
        y = centers_int[..., 1]
        flag = (x >= 0) * (x < w) * (y >= 0) * (y < h)
        flag = flag.reshape(-1, 1).float()
        index = (x + w * y).reshape(-1, 1) * flag
        target = target.reshape([-1, num_points * 4]) / self.offset_std

        target = torch.cat([index, flag, target], dim=-1)
        targets[:target.shape[0]] = target
        return targets

    def gen_offset_map(self, kpts):
        h, w = self.out_shape
        radius = self.radius
        targets = torch.zeros((self.num_points * 4, h, w), dtype=torch.float32)
        kpts = kpts[..., :2]
        for kpt in kpts:
            x, y = kpt.mean(0).int()
            left, right = x - min(x, radius), x + min(w - x, radius + 1)
            top, bottom = y - min(y, radius), y + min(h - y, radius + 1)
            if right <= left or bottom <= top:
                continue
            nx = torch.arange(left, right)
            ny = torch.arange(top, bottom)
            xx, yy = torch.meshgrid([nx, ny], indexing='ij')
            xy = torch.stack((xx, yy), 2)
            offset = kpt.reshape(-1).repeat(len(nx), len(ny), 1) - xy.repeat(1, 1, self.num_points)
            offset = offset.reshape(len(nx), len(ny), self.num_points, 2)
            br_flag = offset >= 0
            tl_flag = offset < 0
            pos_flag = torch.cat([br_flag, tl_flag], dim=-1)
            neg_flag = ~pos_flag
            x_abs = torch.cat([offset.abs(), offset.abs()], dim=-1)
            target = pos_flag * x_abs + neg_flag * x_abs * self.cross_iou_alpha
            target = target[..., [0, 2, 1, 3]]
            target = target.reshape(len(nx), len(ny), self.num_points * 4)
            targets[:, top:bottom, left:right] = target.permute(2, 1, 0) / self.offset_std
        return targets

    def __getitem__(self, index):
        item = self.items[index]
        img = cv2.imread(item['file_path'])
        assert img is not None, self.logger.error(item['file_path'])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        kpts = np.array(item['kpts'], dtype=np.float32).reshape(-1, self.num_points, 3)
        labels = np.array(item['labels'], dtype=np.int64)
        data = {
            "img": TF.to_tensor(img),
            "kpts": torch.from_numpy(kpts),
            "labels": torch.from_numpy(labels),
        }
        if self.transforms is not None:
            data = self.transforms(data)

        self.out_shape = [int(self.img_shape[0] * self.out_scale), int(self.img_shape[1] * self.out_scale)]
        if self.out_scale != 1:
            data["kpts"][..., 0] *= self.out_scale
            data["kpts"][..., 1] *= self.out_scale
            data["scale_factor"][0] *= self.out_scale
            data["scale_factor"][1] *= self.out_scale

        center = data["kpts"].mean(1)
        flag = (center[..., 0] >= 0) & (center[..., 0] < self.out_shape[1]) & \
            (center[..., 1] >= 0) & (center[..., 1] < self.out_shape[0])
        data["kpts"] = data["kpts"][flag][:self.max_objects]
        data["labels"] = data["labels"][flag][:self.max_objects]

        if self.heatmap:
            data["hm"] = self.gen_heatmap(data["kpts"], data["labels"])
            data["offset"] = self.gen_offset(data["kpts"])
            data["seg"] = self.gen_seg(data["kpts"], data["labels"])

        data["file_name"] = DataContainer(item["file_name"], cpu_only=True)
        data["bboxes"] = DataContainer(kpts2bboxes(data["kpts"].float()), stack=False)
        data["kpts"] = DataContainer(data["kpts"].float(), stack=False)
        data["labels"] = DataContainer(data["labels"], stack=False)

        return data
