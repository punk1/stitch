#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-10 19:35:32
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..layers import BaseModule


class BaseHead(BaseModule, ABC):
    """An abstract class representing a :class:`Head`

    All subclasses should overwrite :meth:`compute_loss`, :meth:`post_process` and :meth:`onnx_export`
    """

    @abstractmethod
    def compute_loss(
        self,
        preds: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Training loss compute interface"""
        raise NotImplementedError()

    @abstractmethod
    def post_process(
        self,
        preds: Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inference post processing interface"""
        raise NotImplementedError()

    @abstractmethod
    def onnx_export(
        self,
        preds: Dict[str, Any],
    ) -> Any:
        """Onnx export interface"""
        raise NotImplementedError()
