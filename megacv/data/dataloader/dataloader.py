#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataloader.

Author:     kaizhang
Created at: 2022-04-01 13:20:39
"""

import logging
import random
from functools import cached_property, partial
from typing import Any, Dict, List, Optional, Tuple

import horovod.torch as hvd
import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import T_co

from ...parallel import DataContainer, collate
from ...utils import get_dist_info
from ..builder import DATALOADER
from ..datasets import BaseDataset
from ..samplers import DistributedGroupSampler, GroupSampler, RepeatSampler


@DATALOADER.register_module()
class DataLoader(torch.utils.data.DataLoader):

    """Combines a dataset and a sampler, and provides an iterable over
       the given dataset.

       The :class:`~torch.utils.data.DataLoader` supports both map-style and
       iterable-style datasets with single- or multi-process loading, customizing
       loading order and optional automatic batching (collation) and memory pinning.

       See :py:mod:`torch.utils.data` documentation page for more details.

       Args:
          dataset (Dataset): dataset from which to load the data.
          batch_size (int, optional): how many samples per batch to load
              (default: ``1``).
          shuffle (bool, optional): set to ``True`` to have the data reshuffled
              at every epoch (default: ``False``).
          sampler (Sampler or Iterable, optional): defines the strategy to draw
              samples from the dataset. Can be any ``Iterable`` with ``__len__``
              implemented. If specified, :attr:`shuffle` must not be specified.
          batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
              returns a batch of indices at a time. Mutually exclusive with
              :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
              and :attr:`drop_last`.
          num_workers (int, optional): how many subprocesses to use for data
              loading. ``0`` means that the data will be loaded in the main process.
              (default: ``0``)
          pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
              into CUDA pinned memory before returning them.  If your data elements
              are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
              see the example below.
          drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
              if the dataset size is not divisible by the batch size. If ``False`` and
              the size of dataset is not divisible by the batch size, then the last batch
              will be smaller. (default: ``False``)
          timeout (numeric, optional): if positive, the timeout value for collecting a batch
              from workers. Should always be non-negative. (default: ``0``)
          prefetch_factor (int, optional, keyword-only arg): Number of samples loaded
              in advance by each worker. ``2`` means there will be a total of
              2 * num_workers samples prefetched across all workers. (default: ``2``)
          seed (int): random seed. (default: ``1234``)
          to_cuda (bool): whether to put data to GPU. (default: ``True``)
          start_epoch (int): set start_epoch for batch_sampler. (default ``1``)
          total_epochs (int): set total_epochs for batch_sampler. (default ``1``)
          channels_last (list): keys to convert to channels_last. (default ``None``)
    """

    def __init__(
        self,
        dataset: BaseDataset[T_co],
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
        pin_memory: bool = True,
        drop_last: bool = False,
        timeout: float = 0,
        prefetch_factor: int = 2,
        seed: Optional[int] = 1234,
        to_cuda: bool = True,
        start_epoch: int = 0,
        total_epochs: int = 1,
        channels_last: List[str] = None,
    ):
        rank, world_size = get_dist_info()
        group = hasattr(dataset, 'flag') and len(set(dataset.flag)) > 1
        if dist.is_initialized() or hvd.is_initialized():
            if group:
                sampler = DistributedGroupSampler(dataset, batch_size, world_size, rank, seed)
            else:
                sampler = DistributedSampler(dataset, world_size, rank, shuffle, seed)
        else:
            if group:
                sampler = GroupSampler(dataset, batch_size)
            else:
                sampler = None

        shuffle = shuffle if sampler is None else False
        if seed is not None:
            init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)
        else:
            init_fn = None

        prefetch_factor = None if num_workers == 0 else prefetch_factor
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=batch_size),
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=init_fn,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
        )
        self.logger = logging.getLogger()
        self.total_epochs = total_epochs
        self.channels_last = channels_last
        self.to_cuda = to_cuda
        repeat_sampler = RepeatSampler(self.batch_sampler, start_epoch, total_epochs)
        object.__setattr__(self, 'batch_sampler', repeat_sampler)
        self.reset()

    def reset(self):
        self.epoch = 0
        self.step = 0
        self.data_iter = iter(self)

    def load_data(self, sample, prefix=''):
        if torch.is_tensor(sample) and self.to_cuda:
            return sample.cuda(non_blocking=True)
        elif isinstance(sample, dict):
            res = {}
            for k, v in sample.items():
                v = self.load_data(v, f'{prefix}{k}.')
                if self.channels_last and torch.is_tensor(v) and f'{prefix}{k}' in self.channels_last:
                    v = v.contiguous(memory_format=torch.channels_last)
                res[k] = v
            return res
        elif isinstance(sample, (list, tuple)):
            res = []
            for k, v in enumerate(sample):
                v = self.load_data(v, f'{prefix}{k}.')
                if self.channels_last and torch.is_tensor(v) and f'{prefix}{k}' in self.channels_last:
                    v = v.contiguous(memory_format=torch.channels_last)
                res.append(v)
            return res
        elif isinstance(sample, DataContainer):
            return self.load_data(sample.data[0])
        else:
            return sample

    @cached_property
    def dataset_size(self):
        return len(self.dataset)

    def get_state(self) -> Tuple[int, int]:
        return self.epoch, self.step

    def set_state(self, epoch: int, step: int):
        self.epoch = epoch
        while self.step < step:
            _ = next(self.data_iter)
            self.step += 1
        self.step = step

    def get_batch(self) -> Dict[str, Any]:
        try:
            data = next(self.data_iter)
            self.step += 1
            return self.load_data(data)
        except StopIteration:
            pass
        except Exception as e:
            self.logger.exception(e)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
