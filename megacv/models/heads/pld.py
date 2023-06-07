#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-07 14:07:08
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from megacv.data.visualizer.pld import draw_heatmap, draw_kpts, draw_seg
from megacv.models.builder import HEADS, LOSSES
from megacv.models.heads import BaseHead
from megacv.models.layers import (CBAM, Aggregation, DropBlock2D,
                                  LinearScheduler, ScaledUpsample)
from megacv.models.postprocessing.pld import (get_coarse_kpts, get_fine_kpts,
                                              get_local_maximum,
                                              get_topk_offset, get_topk_points,
                                              kpts2bboxes, match_fine_kpts)


@HEADS.register_module()
class PLDHead(BaseHead):

    def __init__(
        self,
        in_channels: Union[int, List[int]] = 32,
        out_channels: int = 32,
        upsample_ratio: int = 4,
        pred_convs: int = 0,
        kernel_size: int = 5,
        norm_cfg: Dict[str, Any] = {"type": "BN"},
        act_cfg: Dict[str, Any] = {"type": "ReLU"},
        upsample_cfg: Dict[str, Any] = {"mode": "bilinear"},
        use_cbam: bool = True,
        use_dropblock: bool = True,
        block_size: int = 5,
        drop_prob: float = 0.2,
        drop_step: int = 1000,
        pred_cfg: Dict[str, Any] = {
            "hm": {"channels": 5, "act": "sigmoid"},
            "offset": {"channels": 16, "act": "relu"},
        },
        loss_cfg: Dict[str, Any] = {
            "hm": {"type": "GaussianFocalLoss", "loss_weight": 1.0},
            "offset": {"type": "CrossIOULoss", "loss_weight": 1.0},
        },
        kdloss_cfg: Dict[str, Any] = {},
        test_cfg: Dict[str, Any] = {
            "with_topk": True,
            "offset_std": 64,
            "match_radius": 16,
            "center_thr": 0.1,
            "point_thr": 0.01,
            "nms_thr": 0.5,
        },
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.use_dropblock = use_dropblock
        self.use_cbam = use_cbam
        self.upsample_cfg = upsample_cfg
        self.test_cfg = test_cfg

        padding = kernel_size // 2
        self.aggregation = Aggregation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
        )

        if use_dropblock:
            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_prob, block_size=block_size),
                start_value=0.,
                stop_value=drop_prob,
                nr_steps=drop_step,
            )

        if use_cbam:
            self.cbam = CBAM(out_channels, 4)

        self.pred_heads = nn.ModuleDict()
        for name, cfg in pred_cfg.items():
            self.pred_heads[name] = nn.Sequential()
            for _ in range(pred_convs):
                self.pred_heads[name].append(ConvModule(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ))
            if upsample_ratio >= 2:
                self.pred_heads[name].append(ScaledUpsample(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    scale_factor=upsample_ratio,
                    kernel_size=kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    upsample_cfg=upsample_cfg,
                ))
            self.pred_heads[name].append(nn.Conv2d(
                in_channels=out_channels,
                out_channels=cfg["channels"],
                kernel_size=3,
                padding=1,
            ))
            if cfg.get("act") == "relu":
                self.pred_heads[name].append(nn.ReLU(inplace=True))
            elif cfg.get("act") == "softplus":
                self.pred_heads[name].append(nn.Softplus())
            elif cfg.get("act") == "sigmoid":
                self.pred_heads[name].append(nn.Sigmoid())

        self.losses = nn.ModuleDict()
        for name, cfg in loss_cfg.items():
            self.losses[name] = LOSSES.build(cfg)
        for name, cfg in kdloss_cfg.items():
            self.losses[f"kd_{name}"] = LOSSES.build(cfg)

    def compute_kdloss(self, preds: Dict[str, Any], teacher_preds: Dict[str, Any]) -> Dict[str, Any]:
        losses = {
            "kd_hm_loss": self.losses["kd_hm"](preds["hm"], teacher_preds["hm"])
        }
        return losses

    def decode_kpts(self, preds: Dict[str, Any]):
        scores = get_local_maximum(preds["hm"])
        cfg = self.test_cfg
        if cfg.match:
            center_points = get_topk_points(scores[:, cfg.num_points:], topk=cfg.topk, with_cls=True)
            corner_points = get_topk_points(scores[:, :cfg.num_points], topk=cfg.topk, with_cls=False)
        else:
            center_points = get_topk_points(scores, topk=cfg.topk, with_cls=True)
            corner_points = None

        offset = get_topk_offset(center_points, preds["offset"])
        coarse_kpts = get_coarse_kpts(center_points, offset, cfg.offset_std)
        return coarse_kpts, center_points, corner_points

    def compute_loss(self, preds: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        losses = {}
        losses["hm_loss"] = self.losses["hm"](preds["hm"], inputs["hm"])

        index = inputs["offset"][..., 0]
        weight = inputs["offset"][..., 1]
        target = inputs["offset"][..., 2:]
        n, c, h, w = preds["offset"].shape
        offset = preds["offset"].permute(0, 2, 3, 1).reshape(n, -1, c)
        idx = index.unsqueeze(-1).repeat(1, 1, c).long()
        offset = offset.gather(1, idx)
        losses["offset_loss"] = self.losses["offset"](offset, target, weight)

        '''
        offset = preds['offset'].permute(0, 2, 3, 1)
        target = inputs['offset'].permute(0, 2, 3, 1)
        center_hm = inputs['hm'][:, self.test_cfg.num_points:] if self.test_cfg.match else inputs['hm']
        weight = torch.clamp(center_hm.sum(1), min=0, max=1)
        losses["offset_loss"] = self.losses["offset"](offset, target, weight=weight)
        '''

        if "seg" in self.losses:
            losses["seg_loss"] = self.losses["seg"](preds["seg"], inputs["seg"].long())

        if "iou" in self.losses:
            ys = torch.div(index, w, rounding_mode='floor').int().float()
            xs = index - w * ys
            center_points = torch.stack([xs, ys], dim=-1)
            kpts = get_coarse_kpts(center_points, offset, self.test_cfg.offset_std)
            gt_kpts = get_coarse_kpts(center_points, target, self.test_cfg.offset_std)
            bboxes = kpts2bboxes(kpts)
            gt_bboxes = kpts2bboxes(gt_kpts)
            losses["iou_loss"] = self.losses["iou"](bboxes.view(-1, 4), gt_bboxes.view(-1, 4), weight=weight.view(-1))
            # priors = torch.concat([center_points, torch.ones(center_points.shape).cuda() * 2], dim=2)

            # lengths = ((kpts[..., :2] - torch.roll(kpts, 1, 2)[..., :2])**2).sum(3).sqrt()
            # gt_lengths = ((gt_kpts[..., :2] - torch.roll(gt_kpts, 1, 2)[..., :2])**2).sum(3).sqrt()
            # losses["iou_loss"] = self.losses["iou"](lengths, gt_lengths, weight=weight.view(n, c, 1).repeat(1, 1, 4))

        return losses

    def vis(self, preds: Dict[str, Any], inputs: Dict[str, Any], fine_kpts: List[Any], idx: int = 0) -> None:
        img = inputs['img'].detach().cpu().numpy()
        seg = torch.argmax(preds['seg'], dim=1)
        draw_heatmap(img[idx], preds['hm'][idx].detach().cpu().numpy(), "hm.jpg")
        draw_kpts(img[idx], inputs['kpts'][idx].detach().cpu().numpy(), "gt.jpg")
        draw_kpts(img[idx], fine_kpts[idx][..., :12].reshape(-1, 4, 3), "dt.jpg")
        draw_seg(img[idx], seg[idx].detach().cpu().numpy())

    def post_process(self, preds: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        coarse_kpts, center_points, corner_points = self.decode_kpts(preds)
        kpts = get_fine_kpts(center_points.detach().cpu().numpy(),
                             coarse_kpts.detach().cpu().numpy(),
                             self.test_cfg)
        if self.test_cfg.match:
            kpts = match_fine_kpts(corner_points.detach().cpu().numpy(), kpts, self.test_cfg)
        return {"kpts": kpts}

    def onnx_export(self, preds: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.test_cfg.with_kpts:
            coarse_kpts, center_points, corner_points = self.decode_kpts(preds)
            return coarse_kpts, center_points, corner_points
        if self.test_cfg.with_topk:
            scores = get_local_maximum(preds["hm"], self.test_cfg.kernel)
            points = get_topk_points(scores, topk=self.test_cfg.topk)
            offset = preds["offset"].permute(0, 2, 3, 1)
            return points, offset
        else:
            kernel = self.test_cfg.kernel
            pad = (kernel - 1) // 2
            hmax = F.max_pool2d(preds['hm'], (kernel, kernel), stride=1, padding=pad)
            return hmax, preds['hm'], preds["offset"]

    @torch.no_grad()
    def compute_coordinates(self, feats):
        b, _, h, w = feats.shape
        y_loc = torch.tensor([i for i in np.linspace(-1, 1, h).astype(np.float32)], device=feats.device)
        x_loc = torch.tensor([i for i in np.linspace(-1, 1, w).astype(np.float32)], device=feats.device)
        # y_loc = torch.linspace(-1, 1, h, device=feats.device)
        # x_loc = torch.linspace(-1, 1, w, device=feats.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([1, -1, -1])
        x_loc = x_loc.expand([1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 0).expand([b, -1, -1, -1])
        return locations

    def forward(self, inputs: List[torch.Tensor]) -> Dict[str, Any]:
        feats = self.aggregation(inputs)
        if self.use_dropblock:
            self.dropblock.step()
            feats = self.dropblock(feats)
        if self.use_cbam:
            feats = self.cbam(feats)

        preds = {name: head(feats) for name, head in self.pred_heads.items()}
        if 'offset' in preds:
            preds['offset'] = torch.clamp(preds['offset'], 0, 4)
        return preds
