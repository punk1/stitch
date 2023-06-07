#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-13 15:30:16
"""

import collections
import logging
import os
import pickle
from abc import ABC, abstractmethod
from itertools import chain
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed as dist

from ..utils import get_dist_info


def flatten(results: List[Any]) -> Union[List[Any], Dict[str, Any]]:
    elem = results[0]
    elem_type = type(elem)
    if isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({k: flatten([d[k] for d in results]) for k in elem})
        except TypeError:
            return {k: flatten([d[k] for d in results]) for k in elem}
    elif isinstance(elem, collections.abc.Sequence):
        return list(chain(*results))
    elif isinstance(elem, np.ndarray):
        return np.concatenate(results)
    elif isinstance(elem, torch.Tensor):
        return torch.concat([x.cpu() for x in results])
    else:
        return results


class BaseEvaluator(ABC):

    """Evaluator Abstract Interface.

    All subclasses should overwrite :meth:`update` and :meth:`evaluate`

    Args:
        save_dir (str): directory to save evaluation files
        kwargs (dict): default evaluator cfg
    """

    def __init__(self, save_dir, **kwargs):
        self.logger = logging.getLogger()
        self.save_dir = save_dir
        for k, v in kwargs.items():
            setattr(self, k, v)

    def reset(self):
        """Reset evaluator property
        """
        pass

    def collate(self, results: List[Any], dataset_size: int) -> Dict[str, Any]:
        """Collate distributed results
        """
        rank, world_size = get_dist_info()
        if world_size > 1:
            filename = os.path.join(self.save_dir, f'eval_results_{rank}.pkl')
            pickle.dump(results, open(filename, 'wb'))
            dist.barrier()
            if rank == 0:
                multi_results = [results]
                for i in range(1, world_size):
                    filename = os.path.join(self.save_dir, f"eval_results_{i}.pkl")
                    docs = pickle.load(open(filename, "rb"))
                    multi_results.append(docs)

                total_size = len(multi_results[0]) * world_size
                results = [0] * total_size
                for i in range(world_size):
                    results[i:total_size:world_size] = multi_results[i]
                results = results[:dataset_size]

        return flatten(results)

    @abstractmethod
    def update(self, preds: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process one batch outputs.

        Args:
            preds (dict): Model predictions
            inputs (dict): Dataset sample

        Returns:
            Evaluation result.
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, preds: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all results.

        Args:
            preds (dict): Flatten preds, the value size will be exactly same as len(dataset)

        Returns:
            metrics (dict): Evaluation metrics
        """
        raise NotImplementedError()
