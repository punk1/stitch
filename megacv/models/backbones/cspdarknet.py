#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-17 12:24:41
"""

import math

import torch
import torch.nn as nn

from ..builder import BACKBONES
from ..layers import BaseModule

# from mmcv.cnn import Swish


def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # self.act = Swish() if act else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        # self.act = Swish()
        self.act = nn.ReLU()
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPCSP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        # self.act = Swish()
        self.act = nn.ReLU()
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.focus = Conv(c1, c1 * 4, 3, 2, 1, g, act)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # return self.conv(self.focus(x))
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


@BACKBONES.register_module()
class CSPDarknet(BaseModule):

    def __init__(self, depth_multiple=1, width_multiple=1):
        super(CSPDarknet, self).__init__()
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple
        self.model, self.save = self.gen_layers()

    def gen_layers(self):
        config = [
            [-1, 1, Focus, [64, 3]],  # 0-P1/2
            [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
            [-1, 3, BottleneckCSP, [128]],
            [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
            [-1, 9, BottleneckCSP, [256]],
            [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
            [-1, 9, BottleneckCSP, [512]],
            [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
            [-1, 1, SPP, [1024, [5, 9, 13]]],
            [-1, 3, BottleneckCSP, [1024, False]],  # 9
        ]
        layers = []
        save = []
        ch = [3]
        for i, (f, n, m, args) in enumerate(config):
            n = max(round(n * self.depth_multiple), 1)
            if m in [Conv, SPP, Focus, BottleneckCSP]:
                c1, c2 = ch[f], args[0]
                c2 = make_divisible(c2 * self.width_multiple, 8)
                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP]:
                    args.insert(2, n)
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
            else:
                c2 = ch[f]
            m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
            m_.i, m_.f = i, f
            layers.append(m_)
            ch.append(c2)
            if m in [BottleneckCSP]:
                save.append(m_.i)
        return nn.Sequential(*layers), save

    def forward(self, x):
        outs = []
        for m in self.model:
            x = m(x)
            if m.i in self.save:
                outs.append(x)
        return outs
