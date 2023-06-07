#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from megacv.models.builder import HEADS, LOSSES
from megacv.models.heads import BaseHead
from megacv.models.layers import Aggregation


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args=dict()):
        super().__init__()

        self.spatial = args.setdefault('SPATIAL', True)

        self.S = args.setdefault('MD_S', 1)
        self.D = args.setdefault('MD_D', 512)
        self.R = args.setdefault('MD_R', 64)

        self.train_steps = args.setdefault('TRAIN_STEPS', 6)
        self.eval_steps = args.setdefault('EVAL_STEPS', 7)

        self.inv_t = args.setdefault('INV_T', 100)
        self.eta = args.setdefault('ETA', 0.9)

        self.rand_init = args.setdefault('RAND_INIT', True)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    # @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args=dict()):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


class Hamburger(nn.Module):
    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=None,
            act_cfg=None
        )

        self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels,
            ham_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


@HEADS.register_module()
class LightHamHead(BaseHead):
    """Is Attention Better Than Matrix Decomposition?
    This head is the implementation of `HamNet
    <https://arxiv.org/abs/2109.04553>`_.
    Args:
        ham_channels (int): input channels for Hamburger.
        ham_kwargs (int): kwagrs for Ham.

    """

    def __init__(self,
                 num_classes,
                 in_channels=[32, 64, 160, 256],
                 in_index=[1, 2, 3],
                 channels=256,
                 ham_channels=256,
                 dropout_ratio=0.1,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 upsample_cfg=dict(mode="bilinear", align_corners=False),
                 loss_cfg=dict(seg=dict(type="mmdet.CrossEntropyLoss", ignore_index=255)),
                 ham_kwargs=dict(),
                 **kwargs):
        super().__init__()
        self.in_index = in_index
        self.ham_channels = ham_channels
        self.upsample_cfg = upsample_cfg
        self.aggregation = Aggregation(
            in_channels=[in_channels[i] for i in in_index],
            out_channels=ham_channels,
            kernel_size=1,
            padding=0,
            num_convs=0,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
        )
        self.hamburger = Hamburger(ham_channels, ham_kwargs, norm_cfg)
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.align = ConvModule(
            ham_channels,
            channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.losses = nn.ModuleDict()
        for name, cfg in loss_cfg.items():
            self.losses[name] = LOSSES.build(cfg)

    def forward(self, inputs):
        feats = [inputs[i] for i in self.in_index]
        feats = self.aggregation(feats)
        feats = self.hamburger(feats)
        feats = self.align(feats)
        feats = self.dropout(feats)
        output = self.conv_seg(feats)
        return {"seg": output}

    def compute_loss(self, preds, inputs):
        preds['seg'] = F.interpolate(preds['seg'], size=inputs['img'].shape[2:], **self.upsample_cfg)
        target = inputs["seg"].squeeze(1).long()
        losses = {
            "ce_loss": self.losses["seg"](preds["seg"], target),
        }
        return losses

    def post_process(self, preds, inputs):
        preds['seg'] = F.interpolate(preds['seg'], size=inputs['img'].shape[2:], **self.upsample_cfg)
        return {"seg": torch.argmax(preds["seg"], dim=1)}

    def onnx_export(self, preds):
        return torch.argmax(preds["seg"], dim=1)
