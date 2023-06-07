#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-12 11:03:55
"""

from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def denorm(img: np.ndarray) -> np.ndarray:
    """Change normalized image to RGB image.

    Args:
        img (np.ndarray): normalized numpy array with shape C * H * W

    Returns:
        RGB image numpy ndarray
    """
    mean = np.array([123.675, 116.28, 103.53]).reshape([1, 1, 3]) / 255
    std = np.array([58.395, 57.12, 57.375]).reshape([1, 1, 3]) / 255
    img = np.transpose(img, [1, 2, 0])
    img = ((img * std + mean) * 255).astype(np.uint8)
    return img


def draw_heatmap(
    img: np.ndarray,
    heatmap: np.ndarray,
    out: str = "heatmap.jpg",
    normalized: bool = True,
) -> None:
    """Draw heatmap on image.

    Args:
        img (np.ndarray): image numpy array with shape: C * H * W or H * W * C
        heatmap (np.ndarray): heatmap numpy array with shape: N * H * W
        normalized (bool): whether image is normalized
    """
    img = denorm(img) if normalized else img.astype(np.int32)
    corners_heatmap = np.zeros(heatmap.shape[-2:], dtype=np.float32)

    for i in range(heatmap.shape[0]):
        corners_heatmap = np.maximum(corners_heatmap, heatmap[i, ...])

    corners_heatmap = np.tile(corners_heatmap[..., np.newaxis], [1, 1, 3])
    corners_heatmap = (corners_heatmap * 255).astype(np.uint8)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(corners_heatmap)
    plt.subplot(1, 2, 2)
    plt.imshow((img * 0.6 + corners_heatmap * 0.4).astype(np.uint8))
    plt.savefig(out, bbox_inches='tight', pad_inches=1)
    plt.close()


def draw_seg(
    img: np.ndarray,
    seg: np.ndarray,
    out: str = "seg.jpg",
    normalized: bool = True,
    colormap: Dict[int, tuple] = {
        0: (0, 0, 0),        # 其他，
        1: (255, 0, 0),      # 路面，红色
        2: (0, 0, 255),      # 四轮车，蓝色
        3: (128, 128, 105),  # 两轮车，暖灰色
        4: (255, 255, 0),    # 行人，黄色
        5: (0, 255, 0),      # 灌木丛，绿色
        6: (0, 255, 0),      # 路沿，绿色
        7: (252, 230, 201),  # 墙面, 蛋壳色
        8: (252, 230, 201),  # 立柱，蛋壳色
        9: (0, 255, 255),    # 锥形桶，青色
        10: (255, 0, 255),   # 指示牌，粉红
    },
) -> None:
    """Draw segmentation on image.

    Args:
        img (np.ndarray): image numpy array with shape: C * H * W or H * W * C
        seg (np.ndarray): segmentation numpy array with shape: H * W
        normalized (bool): whether image is normalized
    """
    img = denorm(img) if normalized else img.astype(np.int32)
    cmap = np.array(list(colormap.values())).reshape(-1).tolist()
    seg = Image.fromarray(seg.astype(np.uint8), mode='P')
    seg.putpalette(cmap)
    seg = np.array(seg.convert('RGB'))

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(seg)
    plt.subplot(1, 2, 2)
    plt.imshow((img * 0.6 + seg * 0.4).astype(np.uint8))
    plt.savefig(out, bbox_inches='tight', pad_inches=1)
    plt.close()


def draw_kpts(
    img: np.ndarray,
    kpts: np.ndarray,
    out: str = "kpts.jpg",
    normalized: bool = True,
    labels: List[float] = None,
    scores: List[float] = None,
    colormap: Dict[int, tuple] = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (128, 128, 105),
        4: (255, 255, 0),
        5: (0, 255, 0),
    },
) -> np.ndarray:
    """Draw keypoints on image.

    Args:
        img (np.ndarray): image numpy array with shape: C * H * W or H * W * C
        kpts (np.ndarray): keypoints numpy array with shape: N * 4 * 3 or N * 4 * 2
        normalized (bool): whether image is normalized
    """
    img = denorm(img) if normalized else img.astype(np.uint8)
    img = img.copy()
    kpts = kpts.astype(np.int64)
    if labels is not None:
        for k, v in colormap.items():
            selected = [kpt[..., :2] for kpt, c in zip(kpts, labels) if c == k]
            cv2.polylines(img, selected, True, v)
    else:
        cv2.polylines(img, kpts[..., :2], True, (0, 255, 0))

    if scores is None:
        scores = list(range(len(kpts)))
    for score, kpt in zip(scores, kpts):
        cv2.putText(img, '0', kpt[0, :2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img, '1', kpt[1, :2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(img, str(score), kpt[:, :2].mean(0).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(out, img)
    return img


def draw_bboxes(
    img: np.ndarray,
    bboxes: np.ndarray,
    out: str = "bboxes.jpg",
    normalized: bool = True,
    labels: List = None,
    colormap: Dict[int, tuple] = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (128, 128, 105),
        4: (255, 255, 0),
        5: (0, 255, 0),
    },
):
    """Draw bbox on image.

    Args:
        img (np.ndarray): image numpy array with shape: C * H * W or H * W * C
        bboxes (np.ndarray): xyxy numpy array with shape: N * 4
        normalized (bool): whether image is normalized
    """
    img = denorm(img) if normalized else img.astype(np.uint8)
    img = img.copy()
    bboxes = bboxes.astype(np.int64)
    if labels is not None:
        for k, v in colormap.items():
            selected = [bbox for bbox, c in zip(bboxes, labels) if c == k]
            for bbox in selected:
                cv2.rectangle(img, bbox[:2], bbox[2:], v)
    else:
        for bbox in bboxes:
            cv2.rectangle(img, bbox[:2], bbox[2:], (0, 255, 0))
    for i, bbox in enumerate(bboxes):
        cv2.putText(img, str(i), bbox.reshape(2, 2).mean(0).astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(out, img)
    return img
