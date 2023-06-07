# -*- coding: utf-8 -*-
"""
Author:   qichengzuo
Created:  2022-07-27 17:42:14
"""


import torch

from . import bev_height_pool_ext

__all__ = ["bev_height_pool"]


class BEVHeightPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, points, maskes, downsample, pool_method, overlap, interpolate):
        _, _, C, H, W = x.shape
        ctx.saved_shapes = H, W, downsample, pool_method, overlap, interpolate
        out, out_index, out_num = bev_height_pool_ext.bev_height_pool_forward(
            x, points, maskes, downsample, pool_method, overlap, interpolate)
        ctx.mark_non_differentiable(points, maskes, out_index, out_num)

        ctx.save_for_backward(points, maskes, out_index, out_num)

        return out

    @staticmethod
    def backward(ctx, outgrad):
        H, W, downsample, pool_method, overlap, interpolate = ctx.saved_shapes
        points, maskes, out_index, out_num = ctx.saved_tensors
        outgrad = outgrad.contiguous()
        x_grad = bev_height_pool_ext.bev_height_pool_backward(
            outgrad, points, maskes, out_index, out_num, H, W, downsample, pool_method, overlap, interpolate)
        return x_grad, None, None, None, None, None, None


def bev_height_pool(x, points, maskes, downsample, pool_method=1, overlap=0, interpolate=0):
    return BEVHeightPool.apply(x, points, maskes, downsample, pool_method, overlap, interpolate)
