#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-08-11 18:04:00
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models import YOLOXHead as _YOLOXHead
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData

from megacv.models.builder import HEADS, LOSSES
from megacv.models.layers.common import ImplicitA, ImplicitM
from megacv.models.postprocessing.kpt_nms import polygon_nms


@HEADS.register_module()
class YOLOXHead(_YOLOXHead):

    def __init__(self, *args,
                 decoupled=True,
                 num_points=4,
                 input_size=[512, 512],
                 loss_kpt=dict(
                     type="mmdet.SmoothL1Loss",
                     reduction="sum",
                     loss_weight=1.0,
                 ),
                 **kwargs):
        self.decoupled = decoupled
        self.num_points = num_points
        self.kpt_channels = num_points * 2
        super().__init__(*args, **kwargs)
        self.loss_kpt = LOSSES.build(loss_kpt)
        self.use_l1 = False
        featmap_sizes = [[input_size[0] // s, input_size[1] // s] for s in self.strides]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=torch.float32,
            with_stride=True
        )
        self.flatten_priors = torch.cat(mlvl_priors)

    def _build_stacked_convs(self, in_channels):
        """Initialize conv layers of a single level head."""
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = in_channels if i == 0 else self.feat_channels
            stacked_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)

    def _build_predictor_v7(self, in_channels):
        """Initialize predictor layers of a single level head."""
        in_channels = self.feat_channels if self.stacked_convs > 0 else in_channels
        if self.decoupled:
            conv_cls = nn.Sequential(
                ImplicitA(in_channels),
                nn.Conv2d(in_channels, self.num_classes, 1),
                ImplicitM(self.num_classes),
            )
            conv_reg = nn.Sequential(
                ImplicitA(in_channels),
                nn.Conv2d(in_channels, 4, 1),
                ImplicitM(4),
            )
            conv_kpt = nn.Sequential(
                ImplicitA(in_channels),
                nn.Conv2d(in_channels, self.kpt_channels, 1),
                ImplicitM(self.kpt_channels),
            )
            conv_obj = nn.Sequential(
                ImplicitA(in_channels),
                nn.Conv2d(in_channels, 1, 1),
                ImplicitM(1),
            )
            return conv_cls, conv_reg, conv_kpt, conv_obj
        else:
            out_channels = self.num_classes + 4 + self.kpt_channels + 1
            return nn.Sequential(
                ImplicitA(in_channels),
                nn.Conv2d(in_channels, out_channels, 1),
                ImplicitM(out_channels),
            )

    def _build_predictor(self, in_channels):
        """Initialize predictor layers of a single level head."""
        in_channels = self.feat_channels if self.stacked_convs > 0 else in_channels
        if self.decoupled:
            conv_cls = nn.Conv2d(in_channels, self.cls_out_channels, 1)
            conv_reg = nn.Conv2d(in_channels, 4, 1)
            conv_kpt = nn.Conv2d(in_channels, self.kpt_channels, 1)
            conv_obj = nn.Conv2d(in_channels, 1, 1)
            return conv_cls, conv_reg, conv_kpt, conv_obj
        else:
            out_channels = self.num_classes + 4 + self.kpt_channels + 1
            return nn.Conv2d(in_channels, out_channels, 1)

    def _init_layers(self):
        if isinstance(self.in_channels, int):
            self.in_channels = [self.in_channels for _ in self.strides]
        if self.decoupled:
            self.multi_level_cls_convs = nn.ModuleList()
            self.multi_level_reg_convs = nn.ModuleList()
            self.multi_level_conv_cls = nn.ModuleList()
            self.multi_level_conv_reg = nn.ModuleList()
            self.multi_level_conv_kpt = nn.ModuleList()
            self.multi_level_conv_obj = nn.ModuleList()
            for in_channels in self.in_channels:
                self.multi_level_cls_convs.append(self._build_stacked_convs(in_channels))
                self.multi_level_reg_convs.append(self._build_stacked_convs(in_channels))
                conv_cls, conv_reg, conv_kpt, conv_obj = self._build_predictor_v7(in_channels)
                self.multi_level_conv_cls.append(conv_cls)
                self.multi_level_conv_reg.append(conv_reg)
                self.multi_level_conv_kpt.append(conv_kpt)
                self.multi_level_conv_obj.append(conv_obj)
        else:
            self.multi_level_stacked_convs = nn.ModuleList()
            self.multi_level_pred_convs = nn.ModuleList()
            for in_channels in self.in_channels:
                self.multi_level_stacked_convs.append(self._build_stacked_convs(in_channels))
                self.multi_level_pred_convs.append(self._build_predictor_v7(in_channels))

    def init_weights(self):
        super(_YOLOXHead, self).init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        if self.decoupled:
            for conv_cls, conv_obj in zip(self.multi_level_conv_cls, self.multi_level_conv_obj):
                conv_cls[1].bias.data.fill_(bias_init)
                conv_obj[1].bias.data.fill_(bias_init)
        else:
            for conv in self.multi_level_pred_convs:
                conv[1].bias.data.fill_(bias_init)

    def forward_single_decoupled(self, x, cls_convs, reg_convs, conv_cls, conv_reg,
                                 conv_kpt, conv_obj):
        """Forward feature of a single scale level."""
        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        kpt_pred = conv_kpt(reg_feat)
        objectness = conv_obj(reg_feat)

        return cls_score, bbox_pred, kpt_pred, objectness

    def forward_single_coupled(self, x, stackd_convs, pred_convs):
        feat = stackd_convs(x)
        pred = pred_convs(feat)
        cls_score = pred[:, :self.num_classes]
        bbox_pred = pred[:, self.num_classes:self.num_classes + 4]
        kpt_pred = pred[:, self.num_classes + 4:self.num_classes + 4 + self.kpt_channels]
        objectness = pred[:, self.num_classes + 4 + self.kpt_channels:]
        return cls_score, bbox_pred, kpt_pred, objectness

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, Any]:
        if self.decoupled:
            cls_preds, bbox_preds, kpt_preds, obj_preds = multi_apply(self.forward_single_decoupled,
                                                                      feats,
                                                                      self.multi_level_cls_convs,
                                                                      self.multi_level_reg_convs,
                                                                      self.multi_level_conv_cls,
                                                                      self.multi_level_conv_reg,
                                                                      self.multi_level_conv_kpt,
                                                                      self.multi_level_conv_obj)
        else:
            cls_preds, bbox_preds, kpt_preds, obj_preds = multi_apply(self.forward_single_coupled,
                                                                      feats,
                                                                      self.multi_level_stacked_convs,
                                                                      self.multi_level_pred_convs)

        return {"cls": cls_preds, "bbox": bbox_preds, "kpt": kpt_preds, "obj": obj_preds}

    def _kpt_decode(self, priors, kpt_preds, with_sigmoid=False):
        N, C = kpt_preds.shape[:2]
        kpt_preds = kpt_preds.view(N, C, self.num_points, -1)
        kpts = kpt_preds.permute(0, 2, 1, 3)[..., :2] * priors[:, 2:] + priors[:, :2]
        kpts = kpts.permute(0, 2, 1, 3)
        # kpts_score = kpt_preds[..., 2:].sigmoid() if with_sigmoid else kpt_preds[..., 2:]
        # return torch.cat([kpts, kpts_score], -1)
        return kpts

    def _get_kpt_target(self, kpt_target, priors):
        kpt_target = kpt_target.reshape(-1, self.num_points, 2)
        target = (kpt_target[..., :2].permute(1, 0, 2) - priors[:, :2]) / priors[:, 2:]
        # target = torch.cat([target.permute(1, 0, 2), kpt_target[..., 2:]], -1)
        # return target.view(-1, 12)
        return target.permute(1, 0, 2).view(-1, self.kpt_channels)

    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes, decoded_kpts,
                           gt_bboxes, gt_kpts, gt_labels):
        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        decoded_kpts = decoded_kpts.reshape(-1, self.kpt_channels)
        gt_kpts = gt_kpts[..., :2].reshape(-1, self.kpt_channels)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            kpt_target = cls_preds.new_zeros((0, self.kpt_channels))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target, kpt_target,
                    l1_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        scores = cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid()
        pred_instances = InstanceData(
            bboxes=decoded_bboxes, scores=scores.sqrt_(), priors=offset_priors)
        gt_instances = InstanceData(
            bboxes=gt_bboxes, labels=gt_labels
        )
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            gt_instances_ignore=None)

        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_labels = assign_result.labels[pos_inds]

        bbox_target = gt_bboxes[pos_assigned_gt_inds]
        kpt_target = gt_kpts[pos_assigned_gt_inds]
        kpt_target = self._get_kpt_target(kpt_target, priors[pos_inds])

        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(pos_gt_labels, self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target, kpt_target,
                l1_target, num_pos_per_img)

    def get_flatten_feats(self, preds: Dict[str, Any], with_sigmoid=False):
        num_imgs = preds["cls"][0].shape[0]
        flatten_priors = self.flatten_priors.to(preds['cls'][0].device)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_pred in preds["cls"]
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in preds["bbox"]
        ]
        flatten_kpt_preds = [
            kpt_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.kpt_channels)
            for kpt_pred in preds["kpt"]
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in preds["obj"]
        ]
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        if with_sigmoid:
            flatten_cls_preds = flatten_cls_preds.sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_kpt_preds = torch.cat(flatten_kpt_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        if with_sigmoid:
            flatten_objectness = flatten_objectness.sigmoid()
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        flatten_kpts = self._kpt_decode(flatten_priors, flatten_kpt_preds, with_sigmoid)
        return (flatten_priors, flatten_cls_preds, flatten_bbox_preds, flatten_kpt_preds,
                flatten_objectness, flatten_bboxes, flatten_kpts)

    def compute_loss(self, preds: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        num_imgs = preds["cls"][0].shape[0]
        (flatten_priors, flatten_cls_preds, flatten_bbox_preds, flatten_kpt_preds,
         flatten_objectness, flatten_bboxes, flatten_kpts) = self.get_flatten_feats(preds)

        (pos_masks, cls_targets, obj_targets, bbox_targets, kpt_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single,
             flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(),
             flatten_kpts.detach(),
             inputs["bboxes"],
             inputs["kpts"],
             inputs["labels"])

        num_pos = torch.tensor(sum(num_fg_imgs), dtype=torch.float, device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)
        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        kpt_targets = torch.cat(kpt_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        loss_bbox = self.loss_bbox(flatten_bboxes.view(-1, 4)[pos_masks],
                                   bbox_targets) / num_total_samples
        loss_kpt = self.loss_kpt(flatten_kpt_preds.view(-1, self.kpt_channels)[pos_masks],
                                 kpt_targets) / num_total_samples
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        loss_cls = self.loss_cls(flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
                                 cls_targets) / num_total_samples

        loss_dict = dict(loss_cls=loss_cls, loss_bbox=loss_bbox,
                         loss_kpt=loss_kpt, loss_obj=loss_obj)

        if self.use_l1:
            loss_l1 = self.loss_l1(flatten_bbox_preds.view(-1, 4)[pos_masks],
                                   l1_targets) / num_total_samples
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict

    def post_process(self, preds: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        (flatten_priors, flatten_cls_preds, flatten_bbox_preds, flatten_kpt_preds,
         flatten_objectness, flatten_bboxes, flatten_kpts) = self.get_flatten_feats(preds, True)

        max_scores, labels = torch.max(flatten_cls_preds, 2)
        scores = max_scores * flatten_objectness
        num_imgs = scores.shape[0]
        kpts_list = []
        for img_id in range(num_imgs):
            valid_mask = scores[img_id] >= self.test_cfg.score_thr
            score = scores[img_id][valid_mask]
            label = labels[img_id][valid_mask]
            kpts = flatten_kpts[img_id][valid_mask]
            kpts_scores = torch.ones((kpts.shape[0], kpts.shape[1], 1), device=kpts.device)
            kpts = torch.cat([kpts, kpts_scores], dim=-1).view(-1, self.num_points * 3)
            kpts = torch.cat([kpts, label[:, None], score[:, None]], dim=-1)
            kpts = polygon_nms(kpts.detach().cpu().numpy(), self.test_cfg.nms.iou_threshold)
            kpts_list.append(kpts)

        return {"kpts": kpts_list}

    def onnx_export(self, preds: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        (flatten_priors, flatten_cls_preds, flatten_bbox_preds, flatten_kpt_preds,
         flatten_objectness, flatten_bboxes, flatten_kpts) = self.get_flatten_feats(preds, True)

        max_scores, labels = torch.max(flatten_cls_preds, 2)
        scores = max_scores * flatten_objectness
        return scores, labels, flatten_kpts
