#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-07-29 19:06:49
"""

import logging
import math

import torch.nn as nn

from ..builder import BACKBONES
from ..layers import BaseModule
from ..layers.common import (MP, SPP, SPPCSPC, SPPF, ST2CSPA, ST2CSPB,  # noqa
                             ST2CSPC, STCSPA, STCSPB, STCSPC, Bottleneck,
                             BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                             Chuncat, Concat, Contract, Conv, DownC, DWConv,
                             Expand, Focus, Foldcut, Ghost, GhostConv,
                             GhostCSPA, GhostCSPB, GhostCSPC, GhostSPPCSPC,
                             GhostStem, ReOrg, RepBottleneck,
                             RepBottleneckCSPA, RepBottleneckCSPB,
                             RepBottleneckCSPC, RepConv, RepConv_OREPA, RepRes,
                             RepResCSPA, RepResCSPB, RepResCSPC, RepResX,
                             RepResXCSPA, RepResXCSPB, RepResXCSPC, Res,
                             ResCSPA, ResCSPB, ResCSPC, ResX, ResXCSPA,
                             ResXCSPB, ResXCSPC, RobustConv, RobustConv2,
                             Shortcut, Stem, SwinTransformer2Block,
                             SwinTransformerBlock)


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger = logging.getLogger()
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    gd, gw = 1, 1

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            args[j] = None if a == 'None' else a
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC,
                 SPP, SPPF, SPPCSPC, GhostSPPCSPC, Focus, Stem, GhostStem,
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                 Res, ResCSPA, ResCSPB, ResCSPC,
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC,
                 ResX, ResXCSPA, ResXCSPB, ResXCSPC,
                 RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC,
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
                 SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
                 SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC, GhostSPPCSPC,
                     BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                     RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                     ResCSPA, ResCSPB, ResCSPC,
                     RepResCSPA, RepResCSPB, RepResCSPC,
                     ResXCSPA, ResXCSPB, ResXCSPC,
                     RepResXCSPA, RepResXCSPB, RepResXCSPC,
                     GhostCSPA, GhostCSPB, GhostCSPC,
                     STCSPA, STCSPB, STCSPC,
                     ST2CSPA, ST2CSPB, ST2CSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Chuncat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Foldcut:
            c2 = ch[f] // 2
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


@BACKBONES.register_module()
class YOLOv7(BaseModule):

    def __init__(self, cfg, out_indices):
        super().__init__()
        self.out_indices = out_indices
        self.model, self.save = parse_model(cfg, [3])

    def forward(self, x):
        y = []
        outs = []
        for i, m in enumerate(self.model):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if i in self.out_indices:
                outs.append(x)
        return outs
