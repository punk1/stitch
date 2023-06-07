#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-14 15:29:05
"""

import logging

import torch.nn as nn


class OpCounter:

    def __init__(self, multiply_adds: int = 1):
        self.multiply_adds = multiply_adds
        self.logger = logging.getLogger()

    def __call__(self, model, *args, **kwargs):
        flops = []

        def conv_hook(module, inputs, outputs):
            _, _, output_height, output_width = outputs.size()
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups) * self.multiply_adds
            bias_ops = 1 if module.bias is not None else 0
            params = module.out_channels * (kernel_ops + bias_ops)
            flops.append(params * output_height * output_width)

        def linear_hook(module, inputs, outputs):
            weight_ops = module.weight.nelement() * self.multiply_adds
            bias_ops = module.bias.nelement()
            flops.append(weight_ops + bias_ops)

        def bn_hook(module, inputs, outputs):
            flops.append(4 * inputs[0][0].nelement())

        def pool_hook(module, inputs, outputs):
            _, output_channel, output_height, output_width = outputs.size()
            flops.append(output_channel * output_height * output_width * module.kernel_size[0] * module.kernel_size[1])

        def register(net):
            childrens = list(net.children())
            if not childrens:
                if isinstance(net, nn.Conv2d):
                    net.register_forward_hook(conv_hook)
                elif isinstance(net, nn.Linear):
                    net.register_forward_hook(linear_hook)
                elif isinstance(net, nn.BatchNorm2d):
                    net.register_forward_hook(bn_hook)
                elif isinstance(net, (nn.Dropout, nn.ReLU, nn.ReLU6,
                                      nn.Sigmoid, nn.Softmax,
                                      nn.Upsample,
                                      nn.MaxPool2d, nn.AvgPool2d)):
                    self.logger.info(f'{net} is ignored')
                else:
                    self.logger.warning(f'{net} is missed')
            else:
                for c in childrens:
                    register(c)

        register(model)
        model(*args, **kwargs)
        return round(sum(flops) / 1e6, 3)
