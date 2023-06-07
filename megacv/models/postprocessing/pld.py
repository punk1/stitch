#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-12 10:45:03
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .kpt_nms import polygon_nms


def kpts2bboxes(kpts):
    """ Convert kpts(n * 4 * 3) to bboxes xyxy(n * 4)
    """
    xmin = torch.min(kpts[..., 0], dim=-1)[0]
    xmax = torch.max(kpts[..., 0], dim=-1)[0]
    ymin = torch.min(kpts[..., 1], dim=-1)[0]
    ymax = torch.max(kpts[..., 1], dim=-1)[0]
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def kpts2masks(img, kpts, labels):
    kpts = kpts.clone()
    img = torch.zeros(img.shape, dtype=labels.dtype)
    for points, label in zip(kpts, labels):
        x_min = torch.min(kpts[..., 0], dim=-1)[0]
        x_max = torch.max(kpts[..., 0], dim=-1)[0]
        y_min = torch.min(kpts[..., 1], dim=-1)[0]
        y_max = torch.max(kpts[..., 1], dim=-1)[0]
        mask = torch.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=labels.dtype)
        points[:, 0] -= x_min
        points[:, 1] -= y_min
        for i in range(points.shape[0]):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            if p1[1] == p2[1]:
                mask[p1[1], min(p1[0], p2[0]): max(p1[0], p2[0])] += 1
            else:
                k = (p2[0] - p1[0]) / (p2[1] - p1[1])
                b = p1[0] - k * p1[1]
                ymin = min(p1[1], p2[1])
                ymax = max(p1[1], p2[1])
                ymin = max(0, min(ymin, mask.shape[0] - 1))
                ymax = max(0, min(ymax, mask.shape[0] - 1))
                for y in range(int(ymin), int(ymax) + 1):
                    x = int(k * y + b)
                    mask[y, :x + 1] += 1
            img[y_min:y_max + 1, x_min:x_max + 1][mask == 1] = label
    return img


def get_local_maximum(
    heat: torch.Tensor,
    kernel: int = 3,
) -> torch.Tensor:
    """Extract local maximum pixel with given kernel.

    Args:
        heat (Tensor): Target heatmap.
        kernel (int): Kernel size of max pooling. Default: 3.

    Returns:
        heat (Tensor): A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    # keep = (hmax == heat).float()
    keep = 1 - torch.ceil(hmax - heat)
    scores = heat * keep
    return scores


def get_topk_points(
    scores: torch.Tensor,
    topk: int = 20,
    with_cls: bool = False,
) -> torch.Tensor:
    """Get topk points from heatmap

    Args:
        scores (torch.Tensor): Heatmap with shape: N * C * H * W
        topk (int): num points to keep

    Returns:
        Topk points tensor with shape: N * C * topk * 3
    """
    n, c, h, w = scores.size()
    if with_cls:
        topk_scores, topk_inds = torch.topk(scores.view(n, -1), topk)
        topk_classes = topk_inds // (h * w)
        topk_inds = topk_inds % (h * w)
    else:
        topk_scores, topk_inds = torch.topk(scores.view(n, c, -1), topk)

    # topk_ys = torch.div(topk_inds, w, rounding_mode='floor').int().float()
    # topk_xs = (topk_inds % w).int().float()
    topk_ys = topk_inds // w
    topk_xs = topk_inds - w * topk_ys

    if with_cls:
        return torch.stack([topk_xs, topk_ys, topk_scores, topk_classes], dim=-1)
    else:
        return torch.stack([topk_xs, topk_ys, topk_scores], dim=-1)


def get_topk_threshold(
    scores: torch.Tensor,
    topk: int = 20,
    threshold: float = 0.1,
) -> torch.Tensor:
    n, c, h, w = scores.size()
    tensor = scores.view(n, -1)
    mask = (tensor >= threshold)
    coords = torch.nonzero(mask)
    values, indices = tensor[mask].sort(descending=True)
    topk_values = values[:topk]
    topk_coords = coords[indices][:topk]
    return torch.cat([topk_coords, topk_values.reshape(-1, 1)], dim=-1)


def get_topk_offset(
    center_points: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Get offset of center points

    Args:
        center_points (torch.Tensor): Center points with shape N * topk * 3
        offset (torch.Tensor): Offset preds with shape N * 16 * H * W

    Returns:
        selcted offset tensor with shape N * topk * 16
    """
    n, c, h, w = offset.shape
    valid = center_points[:, :, 1] * w + center_points[:, :, 0]
    offset = offset.permute(0, 2, 3, 1).reshape(n, -1, c)
    index = valid.unsqueeze(-1).repeat(1, 1, c).long()
    selected = offset.gather(1, index)
    return selected


def get_seg_status(
    center_points: torch.Tensor,
    status: torch.Tensor
):
    """Get status with seg pred
    """
    status = get_topk_offset(center_points, status)
    status = torch.argmax(status, dim=-1)
    status -= 1
    status[status < 0] = 0
    return status


def get_coarse_kpts(
    center_points: torch.Tensor,
    center_offset: torch.Tensor,
    offset_std: float,
) -> torch.Tensor:
    """Get coarse kpts

    Args:
        center_points (torch.Tensor): tensor shape N * topk * 3
        center_offset (torch.Tensor): tensor shape N * topk * 16
        offset_std (float): offset std

    Returns:
        coarse kpts with shape N * topk * 4 * 2
    """
    bs, topk, _ = center_offset.shape
    reg = center_offset.reshape(bs, topk, -1, 2, 2)
    pts, inds = reg.max(dim=-1)
    # pts[inds == 1] *= -1
    pts = pts - 2 * pts * inds
    kpts = center_points[..., :2].unsqueeze(2) + pts * offset_std
    return kpts


def match(
    point: np.ndarray,
    pts: np.ndarray,
    radius: int,
    min_dis: float = 0,
) -> Tuple[np.ndarray, int]:
    """Find point in pts closest to point

    Args:
        point (np.ndarray): target point
        pts (np.ndarray): available points
        radius (int): the max match radius
        min_dis (float): the min distinance

    Returns:
        matched point and match flag
    """
    x, y = point[:2]
    valid = (pts[:, 0] >= x - radius) & (pts[:, 0] <= x + radius) & \
            (pts[:, 1] >= y - radius) & (pts[:, 1] <= y + radius)

    if min_dis > 0:
        invalid = (pts[:, 0] > x - min_dis) & (pts[:, 0] < x + min_dis) & \
                  (pts[:, 1] > y - min_dis) & (pts[:, 1] < y + min_dis)
        valid = valid & (1 - invalid).astype(np.bool)

    _pts = pts[valid, :].copy()
    if _pts.shape[0] <= 0:
        return np.array([x, y, 0]).astype(np.float32), 0

    index = np.argmax(_pts[:, -1])
    return _pts[index], 1


def get_fine_kpts(
    center_points: np.ndarray,
    coarse_kpts: np.ndarray,
    cfg: Dict[str, Any],
) -> List[np.ndarray]:
    """Get fine kpts

    Args:
        center_points: N * topk * 4
        coarse_kpts: N * topk * num_points * 2

    Returns:
        list of kpts with shape N * 14
    """
    n, topk = center_points.shape[:2]
    rets = []
    for i in range(n):
        center_point = center_points[i]
        valid = center_point[:, 2] >= cfg.center_thr
        scores = center_point[:, 2][valid]
        classes = center_point[:, 3][valid]
        kpts = np.zeros((scores.shape[0], cfg.num_points, 3))
        kpts[..., :2] = coarse_kpts[i, valid]
        kpts = kpts.reshape(-1, cfg.num_points * 3)
        kpts = np.hstack([kpts, classes[:, np.newaxis], scores[:, np.newaxis]])
        rets.append(polygon_nms(kpts, cfg.nms_thr))
    return rets


def match_fine_kpts(
    corner_points: np.ndarray,
    fine_kpts: List[np.ndarray],
    cfg: Dict[str, Any],
) -> List[np.ndarray]:
    rets = []
    for i, kpts in enumerate(fine_kpts):
        if not kpts.shape[0]:
            rets.append(kpts)
            continue

        index = []
        extra = kpts[:, 12:]
        kpts = kpts[:, :12].reshape(-1, cfg.num_points, 3)
        for j in range(kpts.shape[0]):
            kpt = kpts[j]
            match_count = 0
            for k in range(cfg.num_points):
                valid = corner_points[i, k, :, 2] >= cfg.point_thr
                pts = corner_points[i, k, valid]
                fine_kpt, flag = match(kpt[k], pts, cfg.match_radius)
                kpts[j, k] = fine_kpt
                match_count += flag
            if match_count >= 1:
                index.append(j)

        kpts = kpts.reshape(-1, cfg.num_points * 3)
        kpts = np.hstack([kpts, extra])
        rets.append(kpts)
    return rets
