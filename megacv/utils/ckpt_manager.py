#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:     kaizhang
Created at: 2022-03-31 15:09:26
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer

from .dist_utils import master_only

logger = logging.getLogger()


class CheckpointManager:
    """CheckpointManager.

    Args:
        save_dir (str): directory to save checkpoint.
        model (nn.Module|nn.parallel.DistributedDataParallel): model to save.
        optimizer (Optimizer): optimizer to save.
        lr_scheduler (Any): learning rate scheduler.
        scaler (Any): the amp grad scaler.
        max_keep (int): maximum checkpoint to keep.
        pretrained (str or dict): path of pretrained model or submodule pretrained model.
        resume (bool): whether to resume, Default: False.
        to_cuda (bool): whether to remap storage locations to cuda, Default: False.
    """

    def __init__(
        self,
        save_dir: str = '',
        model: Union[nn.Module, nn.parallel.DistributedDataParallel, nn.DataParallel] = None,
        optimizer: Union[Optimizer, None] = None,
        lr_scheduler: Any = None,
        scaler: Any = None,
        max_keep: Optional[int] = None,
        pretrained: Optional[Union[str, Dict[str, str]]] = None,
        resume: Optional[bool] = False,
        to_cuda: Optional[bool] = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.resume = resume
        self.pretrained = pretrained
        self.to_cuda = to_cuda
        self.save_dir = Path(save_dir)
        self.last_file = self.save_dir / "ckpt-last.pth"
        self.best_file = self.save_dir / "ckpt-best.pth"
        self.checkpoint = {}
        self.reload()

    def reload(self):
        if self.resume and os.path.exists(self.last_file):
            self.checkpoint = self.load_ckpt(self.last_file, to_cuda=self.to_cuda)
        elif self.pretrained:
            if isinstance(self.pretrained, str):
                assert os.path.exists(self.pretrained), f"{self.pretrained} not exists"
                self.checkpoint = self.load_ckpt(self.pretrained, to_cuda=self.to_cuda)
            elif isinstance(self.pretrained, dict):
                for submodule, filename in self.pretrained.items():
                    assert os.path.exists(filename), f"{filename} not exists"
                    ckpt = self.load_ckpt(filename, to_cuda=self.to_cuda)
                    for k, v in ckpt.items():
                        self.checkpoint[f'{submodule}.{k}'] = v
            else:
                raise ValueError(f"expect pretrained to be str or dict, but got {type(self.pretrained)}")

    def get_last_ckpt(self) -> str:
        return (self.save_dir / "ckpt-last.pth").resolve()

    @staticmethod
    def save_best_ckpt(ckpt_file: str) -> None:
        ckpt_file = Path(ckpt_file)
        best_file = ckpt_file.parent / "ckpt_best.pth"
        if best_file.exists() or best_file.is_symlink():
            best_file.unlink()
        best_file.symlink_to(ckpt_file.name)

    @staticmethod
    def get_ckpt_epoch(name: str) -> int:
        res = re.match(r"ckpt-(\d+)-(\d+).pth", name)
        return int(res.groups()[0]) if res else 0

    @staticmethod
    def load_ckpt(ckpt_path: str, to_cuda: Optional[bool] = False) -> Dict[str, Any]:
        """Load checkpoint.

        Args:
            ckpt_path (str): Checkpoint path.
            to_cuda (bool, optional): Whether to remap storage locations to cuda, Default: False.
        """
        ckpt_path = Path(ckpt_path).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"File {ckpt_path} does not exist.")

        def _map_location(storage, _):
            if to_cuda:
                device = torch.cuda.current_device()
                return storage.cuda(device)
            return storage

        logger.info("Load ckpt %s", ckpt_path)
        return torch.load(ckpt_path, map_location=_map_location)

    @master_only
    def save_ckpt(
        self,
        epoch: int,
        step: int,
        save_best: Optional[bool] = False,
        save_last: Optional[bool] = True,
    ) -> None:
        """Save checkpoint.

        Args:
            epoch (int): Epoch id.
            save_best (bool): If `True`, checkpoint name will be `ckpt-best.pth`
                and overwrite previous file. Default: False.
            save_last (bool): If `True`, checkpoint name will be `ckpt-last.pth`
                and overwrite previous file. Default: True.
        """
        state = {
            "epoch": epoch,
            "step": step,
            "model": self.model.state_dict(),
        }

        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()

        ckpt_file = f"ckpt-{epoch:03d}-{step:05d}.pth"
        torch.save(state, self.save_dir / ckpt_file)
        if save_last:
            if self.last_file.exists() or self.last_file.is_symlink():
                self.last_file.unlink()
            self.last_file.symlink_to(ckpt_file)
        if save_best:
            if self.best_file.exists() or self.best_file.is_symlink():
                self.best_file.unlink()
            self.best_file.symlink_to(ckpt_file)

        if self.max_keep is not None:
            files = sorted(self.save_dir.iterdir(), reverse=True)
            files = [f for f in files if re.match(r"ckpt-(\d+)-(\d+).pth", f.name)]
            for f in files[self.max_keep:]:
                if not (self.best_file.exists() and self.best_file.samefile(f)):
                    f.unlink()

        logger.info("Save checkpoint done: %s", ckpt_file)

    def load_model(self):
        if not self.resume and self.pretrained is None:
            return logger.info("Training from scratch. Skip load model")

        ckpt = self.checkpoint.get("model", self.checkpoint)
        miss_key, unexpect_key = self.model.load_state_dict(ckpt, strict=False)
        logger.info("Miss keys: %s", miss_key)
        logger.info("Unexpect keys: %s", unexpect_key)

    def load_scaler(self):
        if not self.resume:
            return logger.info("Training from scratch. Skip load scaler")

        if "scaler" in self.checkpoint:
            self.scaler.load_state_dict(self.checkpoint["scaler"])
            logger.info("Load scaler done")
        else:
            logger.warning("Key `scaler` not in checkpoint")

    def load_optimizer(self):
        if not self.resume:
            return logger.info("Training from scratch. Skip load optimizer")

        if "optimizer" in self.checkpoint:
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])
            logger.info("Load optimizer done")
        else:
            logger.warning("Key `optimizer` not in checkpoint")

    def load_lr_scheduler(self):
        if not self.resume:
            return logger.info("Training from scratch. Skip load lr_scheduler")

        if "lr_scheduler" in self.checkpoint:
            self.lr_scheduler.load_state_dict(self.checkpoint["lr_scheduler"])
            logger.info("Load lr_scheduler done")
        else:
            logger.warning("Key `lr_scheduler` not in checkpoint")

    def load_step(self):
        if not self.resume:
            logger.info("Training from scratch. Skip load step")
            return 0

        return self.checkpoint.get("step", 0)

    def load_epoch(self):
        if not self.resume:
            logger.info("Training from scratch. Skip load epoch")
            return 0

        return self.checkpoint.get("epoch", 0)
