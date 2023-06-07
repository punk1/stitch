#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from mmcv.ops import nms
from shapely.geometry import Polygon


def _is_intersect(edgeA, edgeB):
    def determinant(v1, v2, v3, v4):
        return v1 * v4 - v2 * v3
    a, b = edgeA
    c, d = edgeB
    delta = determinant(b[0] - a[0], c[0] - d[0], b[1] - a[1], c[1] - d[1])
    if delta <= 1e-6 and delta >= 1e-6:
        return False
    if delta == 0:
        return False
    miu = determinant(c[0] - a[0], c[0] - d[0], c[1] - a[1], c[1] - d[1]) / delta
    if miu > 1 or miu < 0:
        return False
    miu = determinant(b[0] - a[0], c[0] - a[0], b[1] - a[1], c[1] - a[1]) / delta
    if miu > 1 or miu < 0:
        return False
    return True


def polygon_nms(kpts, thr=0.5):
    kpts = np.array(sorted(kpts.tolist(), key=lambda x: x[-1], reverse=True))
    n = kpts.shape[0]
    if n == 0:
        return kpts
    probs = kpts[:, 12:]
    kpt_probs = kpts[:, :12].reshape(n, 4, 3)[..., 2]
    kpts = kpts[:, :12].reshape(n, 4, 3)[:, :, :2]

    # remove keypoint that have same points or have intersect lines
    index = []
    for i in range(n):
        is_valid = True
        # remove that of same points
        for j in range(4):
            xj, yj = kpts[i, j]
            for k in range(j + 1, 4):
                xk, yk = kpts[i, k]
                if (xj - xk) ** 2 + (yj - yk) ** 2 <= 16:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            continue
        # remove that having intersect lines
        # line 0->1, line 2->3 do not intersect and line 1->2, line 0->3 do not intersect
        if _is_intersect((kpts[i, 0], kpts[i, 1]), (kpts[i, 2], kpts[i, 3])) or \
                _is_intersect((kpts[i, 1], kpts[i, 2]), (kpts[i, 0], kpts[i, 3])):
            is_valid = False

        if is_valid:
            index.append(i)
    kpts = kpts[index, :, :]
    n = kpts.shape[0]
    # to poly
    polys = []
    for i in range(n):
        polys.append(Polygon(kpts[i]).convex_hull)
    invalids = [False] * n
    ret = []
    for i in range(n):
        if invalids[i]:
            continue
        polya = polys[i]
        for j in range(i + 1, n):
            if invalids[j]:
                continue
            # cannot share common point
            share_common_point = False
            for ii in range(4):
                ax, ay = kpts[i, ii]
                if ax < 0 or ay < 0:
                    continue
                for jj in range(4):
                    bx, by = kpts[j, jj]
                    if bx < 0 or by < 0:
                        continue
                    if (ax - bx) ** 2 + (ay - by) ** 2 <= 16:
                        # common point
                        share_common_point = True
                        break
                if share_common_point:
                    break
            # if share_common_point:
            #     invalids[j] = True
            #     continue
            polyb = polys[j]
            inter_area = polya.intersection(polyb).area
            # union_area = polya.area + polyb.area - inter_area
            union_area = max(min(polya.area, polyb.area), 1)
            iou = inter_area / union_area
            if iou > thr:
                invalids[j] = True
            # if iou > 0.1 and share_common_point:
            #     invalids[j] = True

        kpt = np.hstack([kpts[i], kpt_probs[i][:, np.newaxis]])
        ret.append(np.hstack([kpt.reshape(-1), probs[i, :]]))

    return np.array(ret) if ret else np.zeros((0, kpts.shape[1]))


def multiclass_nms_for_kpt(multi_bboxes, multi_kpts, multi_scores, score_thr, nms_cfg, max_num=-1):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, kpts, labels = [], [], []
    for i in range(1, num_classes):
        cls_inds = multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        # get kpts
        _kpts = multi_kpts[cls_inds, :]
        _scores = multi_scores[cls_inds, i]
        cls_kpts = torch.cat([_kpts, _scores[:, None]], dim=1)
        # cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        # cls_dets, inds = nms(cls_dets, **nms_cfg)
        cls_dets, inds = nms(_bboxes, _scores, **nms_cfg)
        cls_labels = multi_bboxes.new_full(
            (cls_dets.shape[0], ), i - 1, dtype=torch.long)
        bboxes.append(cls_dets)
        kpts.append(cls_kpts[inds])
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        kpts = torch.cat(kpts)
        labels = torch.cat(labels)
        # if bboxes.shape[0] > max_num:
        _, inds = bboxes[:, -1].sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        kpts = kpts[inds]
        labels = labels[inds]
        # Add one step for kpt iou condition
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        kpts = multi_kpts.new_zeros((0, 9))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
    return kpts
