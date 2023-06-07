#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-06 17:05:08
"""

import collections
import inspect
import itertools
import logging
import math
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import horovod.torch as hvd
import numpy as np
import psutil
import pynvml
import torch
import torch.distributed as dist
import wandb
from prettytable import PrettyTable
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from ..data.builder import DATALOADER, DATASETS
from ..models.builder import MODELS
from ..optim.builder import LR_SCHEDULER, OPTIMIZER
from ..parallel import DistributedDataParallel
from ..utils import (CheckpointManager, EMAModel, FileManager, Stopwatch,
                     get_dist_info, log_reset)
from .builder import TRAINER

logger = logging.getLogger()


@TRAINER.register_module()
class Trainer:

    """MegaCV default trainer

    Args:
        cfg (dict): complete parameters in yaml
        lr_per_gpu (float): learning rate per gpu, Default: ``1e-3``
        lr_total (float): total learning rate, Default: ``None``
        max_keep (int): max checkpoint to keep, Default: ``50``
        total_epochs (int): training epochs. Default: ``1``
        summary_step (int): logging frequency, Default: ``20``
        accumulation_step (int): update optimizer params after accumulation steps, Default: ``1``
        save_epoch (int): save ckpt by epoch, Default: ``1``
        save_step (int): save ckpt by step, Default: ``None``
        clip_gradient (int): the maximum gradient, Default: ``35``
        use_fp16 (bool): whether to use fp16 training, Default: ``False``
        use_syncbn (bool): whether to use sync batchnorm, Default: ``False``
        use_profile (bool): whether to enable profiler, Default: ``False``
        use_deterministic (bool): whether to enable deterministic algorithms, Default: ``False``
        find_unused_parameters (bool): DDP will analyzes the output from the local model
            if `find_unused_parameters` set to True, Default: ``False``
        seed (int): random seed, Default: ``1234``
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        lr_per_gpu: float = 1e-3,
        lr_total: float = None,
        max_keep: int = 50,
        total_epochs: int = 1,
        summary_step: int = 20,
        accumulation_step: int = 1,
        save_epoch: int = 1,
        save_step: int = None,
        clip_gradient: int = 35,
        use_fp16: bool = False,
        use_syncbn: bool = False,
        global_syncbn: bool = False,
        use_profile: bool = False,
        use_deterministic: bool = False,
        use_benchmark: bool = True,
        detect_anomaly: bool = False,
        find_unused_parameters: bool = False,
        seed: int = 1234,
        **kwargs,
    ):
        torch.cuda.empty_cache()
        pynvml.nvmlInit()

        self.cfg = cfg
        self.task_name = cfg.task_name
        self.total_epochs = total_epochs
        self.max_keep = max_keep
        self.summary_step = summary_step
        self.accumulation_step = accumulation_step
        self.save_epoch = save_epoch
        self.save_step = save_step
        self.clip_gradient = clip_gradient
        self.lr_per_gpu = lr_per_gpu
        self.lr_total = lr_total
        self.use_fp16 = use_fp16
        self.use_syncbn = use_syncbn
        self.global_syncbn = global_syncbn
        self.use_profile = use_profile
        self.use_deterministic = use_deterministic
        self.use_benchmark = use_benchmark
        self.detect_anomaly = detect_anomaly
        self.find_unused_parameters = find_unused_parameters
        self.seed = seed

        self.file_manager = FileManager(cfg)
        self.work_dir = Path(self.file_manager.work_dir).absolute()
        self.current_device = torch.cuda.current_device()
        self.rank, self.num_gpus = get_dist_info()
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
        self.stopwatch = Stopwatch(False)

        self.enable_deterministic()
        self.build_components()
        if self.use_profile:
            self.enable_profiler()

    def enable_deterministic(self) -> None:
        torch.autograd.set_detect_anomaly(self.detect_anomaly)
        torch.use_deterministic_algorithms(self.use_deterministic)
        torch.backends.cudnn.benchmark = self.use_benchmark
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.set_float32_matmul_precision('high')

    def enable_profiler(self) -> None:
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                skip_first=50,
                wait=30,
                warmup=20,
                active=20,
                repeat=1,
            ),
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.file_manager.summary_dir)
        )

    def freeze_model(self) -> None:
        for name, m in self.model.named_modules():
            freezes = [self.cfg.freeze] if isinstance(self.cfg.freeze, str) else self.cfg.freeze
            if any([name.startswith(x) for x in freezes]):
                logger.info(f"freeze module: {name}")
                for p in m.parameters():
                    p.requires_grad = False
                if isinstance(m, _BatchNorm):
                    m.eval()

    def enable_syncbn(self) -> None:
        if self.cfg.horovod or self.global_syncbn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        else:
            group_size = min(torch.cuda.device_count(), self.num_gpus)
            num_groups = self.num_gpus // group_size
            global_process_groups = []
            for i in range(num_groups):
                start_id = i * group_size
                group_ids = [start_id + i for i in range(group_size)]
                global_process_groups.append(dist.new_group(ranks=group_ids, backend="nccl"))
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model,
                global_process_groups[self.rank // group_size]
            )

    def enable_dist(self) -> None:
        if not self.cfg.horovod:
            self.model = DistributedDataParallel(
                module=self.model,
                device_ids=[self.current_device],
                find_unused_parameters=self.find_unused_parameters,
            )
        else:
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)

    def build_model(self) -> None:
        self.model = MODELS.build(self.cfg.model)
        self.model = self.model.cuda(device=self.current_device)
        self.model_inputs = inspect.signature(self.model.forward).parameters
        if self.cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.cfg.freeze:
            self.freeze_model()

        if self.use_syncbn:
            self.enable_syncbn()

        self.model_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        logger.info(f"model params: {len(list(self.model.parameters()))}, trainable: {len(self.model_params)}")

        if self.num_gpus > 1:
            self.enable_dist()

        self.ema_model = EMAModel(self.model, **self.cfg.ema)

    def get_params(self, paramwise_cfg, named_params) -> List[Parameter]:
        return [x[1] for x in named_params]

    def build_optimizer(self) -> None:
        self.cfg.optimizer.lr = self.lr_total or self.lr_per_gpu * self.num_gpus
        self.cfg.lr_scheduler.steps_per_epoch = self.steps_per_epoch

        named_params = list(filter(lambda p: p[1].requires_grad, self.model.named_parameters()))
        paramwise_cfg = self.cfg.optimizer.pop('paramwise_cfg', None)
        lr_mults = self.cfg.get('lr_mults', defaultdict())
        if paramwise_cfg:
            params = self.get_params(paramwise_cfg, named_params)
        else:
            keys = [x[0] for x in named_params]
            lrs = [self.cfg.optimizer.lr] * len(keys)
            for key, mult in lr_mults.items():
                lrs = [(lr * mult if name.startswith(key) else lr) for name, lr in zip(keys, lrs)]
            params = [{'params': y[1], 'lr':x} for x, y in zip(lrs, named_params)]
        self.optimizer = OPTIMIZER.build(self.cfg.optimizer, default_args={"params": params})

        if self.cfg.horovod:
            backward_passes_per_step = len(self.dataloader) * self.accumulation_step
            self.optimizer = hvd.DistributedOptimizer(self.optimizer,
                                                      named_parameters=self.model.named_parameters(),
                                                      backward_passes_per_step=backward_passes_per_step)

        if self.cfg.lr_scheduler.type == 'MegaOneCycleLR':
            max_lrs = [self.cfg.lr_scheduler.max_lr] * len(lrs)
            for key, mult in lr_mults.items():
                max_lrs = [(lr * mult if name.startswith(key) else lr) for name, lr in zip(keys, max_lrs)]
            self.cfg.lr_scheduler.max_lr = max_lrs
        self.lr_scheduler = LR_SCHEDULER.build(self.cfg.lr_scheduler, default_args={"optimizer": self.optimizer})

    def build_dataloader(self, cfg, start_epoch=0):
        dataset = DATASETS.build(cfg.dataset, default_args={"mode": "train"})
        default_args = {
            "dataset": dataset,
            "seed": self.seed,
            "shuffle": True,
            "start_epoch": start_epoch,
            "total_epochs": math.inf,
        }
        return DATALOADER.build(cfg.dataloader, default_args=default_args)

    def build_components(self) -> None:
        start_epoch = 0
        if self.cfg.resume:
            ckpt_path = CheckpointManager(self.file_manager.ckpt_dir).get_last_ckpt()
            ckpt = CheckpointManager.load_ckpt(ckpt_path)
            start_epoch = ckpt.get("epoch", 0)

        self.dataloader = {}
        if self.cfg.task_configs:
            if isinstance(self.cfg.batch_size, int):
                self.cfg.batch_size = [self.cfg.batch_size] * len(self.cfg.task_configs)
            for (task_name, cfg), batch_size in zip(self.cfg.task_configs.items(), self.cfg.batch_size):
                cfg.dataloader.batch_size = batch_size
                self.dataloader[task_name] = self.build_dataloader(cfg, start_epoch)
        else:
            self.dataloader[self.cfg.task_name] = self.build_dataloader(self.cfg, start_epoch)

        self.dataset_size = [x.dataset_size for x in self.dataloader.values()]
        self.batch_per_gpu = [x.batch_size for x in self.dataloader.values()]
        self.steps_per_epoch = max([len(x) for x in self.dataloader.values()])
        self.total_steps = self.total_epochs * self.steps_per_epoch

        self.build_model()
        self.build_optimizer()

        if self.rank == 0:
            self.writer = SummaryWriter(self.file_manager.summary_dir)

        self.ckpt_manager = CheckpointManager(
            save_dir=self.file_manager.ckpt_dir,
            model=self.ema_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            scaler=self.scaler,
            resume=self.cfg.resume,
            pretrained=self.cfg.pretrained,
            max_keep=self.max_keep,
            to_cuda=True,
        )
        self.load_ckpt()

    def load_ckpt(self) -> None:
        self.ckpt_manager.load_model()
        self.ckpt_manager.load_optimizer()
        self.ckpt_manager.load_lr_scheduler()
        self.ckpt_manager.load_scaler()
        self.start_epoch = self.ckpt_manager.load_epoch()
        self.start_step = self.ckpt_manager.load_step()
        self.step_idx = self.start_step
        if self.start_step == self.steps_per_epoch - 1:
            self.start_epoch += 1
            self.step_idx = self.start_step = 0

    def parse_losses(self, losses: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_vars = {}
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                loss_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                loss_vars[loss_name] = sum(loss.mean() for loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(value for key, value in loss_vars.items() if "loss" in key)
        return loss, loss_vars

    def clip_grads(self, params: Iterator[Parameter]) -> None:
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad_norm_(params, max_norm=self.clip_gradient, norm_type=2.0)

        raise ValueError("Clip Gradient Params Size <=0 !!!")

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        preds = self.model(inputs) if len(self.model_inputs) == 1 else self.model(**inputs)
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(preds, inputs)
        else:
            return preds

    def train_one_batch(self, inputs: Dict[str, Any], retain_graph=False) -> Dict[str, torch.Tensor]:
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            losses = self.forward(inputs)
        self.time_cost["model_cost"].append(self.stopwatch.toc2())

        loss, loss_vars = self.parse_losses(losses)
        loss = loss / self.accumulation_step
        if self.use_fp16:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)
        self.time_cost["backward_cost"].append(self.stopwatch.toc2())
        return loss_vars

    def train_one_step(self, epoch: int, step: int) -> None:
        self.stopwatch.tic()
        self.loss_vars = collections.defaultdict(float)
        flag = self.global_step % self.accumulation_step == 0
        mycontext = nullcontext if flag or not hasattr(self.model, 'no_sync') else self.model.no_sync
        with mycontext():
            task_num = len(self.dataloader)
            for i, task_name in enumerate(self.dataloader):
                inputs = self.dataloader[task_name].get_batch()
                if inputs is None:
                    continue
                if self.cfg.model.type == 'MultiDetector':
                    inputs["task"] = task_name
                self.time_cost["data_cost"].append(self.stopwatch.toc2())
                losses = self.train_one_batch(inputs, i < task_num)
                if task_num > 1:
                    losses = {f"{task_name}_{k}": v for k, v in losses.items()}
                for k, v in losses.items():
                    self.loss_vars[k] += v
            self.loss_vars["loss"] = sum(value for key, value in self.loss_vars.items() if key != "loss")

        if not self.cfg.horovod:
            if dist.is_initialized() and dist.get_world_size() > 1:
                for loss_name, loss_value in self.loss_vars.items():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
                    self.loss_vars[loss_name] = loss_value.item()
                self.time_cost["backward_cost"].append(self.stopwatch.toc2())

            if self.global_step % self.accumulation_step == 0:
                if self.clip_gradient > 0:
                    if self.use_fp16:
                        self.scaler.unscale_(self.optimizer)
                    self.clip_grads(self.model_params)
                    self.time_cost["clipgrad_cost"].append(self.stopwatch.toc2())

                if self.use_fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.time_cost["optimizer_cost"].append(self.stopwatch.toc2())
        else:
            self.optimizer.synchronize()
            if self.clip_gradient > 0:
                if self.use_fp16:
                    self.scaler.unscale_(self.optimizer)
                self.clip_grads(self.model_params)
                self.time_cost["clipgrad_cost"].append(self.stopwatch.toc2())

            if self.use_fp16:
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with self.optimizer.skip_synchronize():
                    self.optimizer.step()
            self.optimizer.zero_grad()
            self.time_cost["optimizer_cost"].append(self.stopwatch.toc2())

        self.ema_model.step()
        self.lr_scheduler.step()
        if self.use_profile:
            self.profiler.step()

    def save_model(self, epoch, step):
        self.ckpt_manager.save_ckpt(epoch=epoch, step=step)

    def train_one_epoch(self, epoch: int) -> None:
        for dataloader in self.dataloader.values():
            dataloader.set_state(epoch, self.start_step)

        for self.step_idx in range(self.start_step, self.steps_per_epoch):
            self.global_step = epoch * self.steps_per_epoch + self.step_idx + 1
            self.train_one_step(epoch, self.step_idx)
            if self.global_step % self.summary_step == 0:
                logger.info(self)
                self.time_cost = defaultdict(list)
            if self.save_step and self.global_step % self.save_step == 0 and self.rank == 0:
                self.save_model(epoch, self.step_idx)

        if not self.save_step and (epoch + 1) % self.save_epoch == 0 and self.rank == 0:
            self.save_model(epoch, self.step_idx)

        self.start_step = 0
        self.step_idx = 0

    def train(self) -> None:
        log_reset(logger)
        self.time_anchor = time.time()
        self.time_cost = defaultdict(list)
        self.model.train()
        logger.info(f"Start training epoch: {self.start_epoch}, step: {self.start_step}")
        if self.use_profile:
            logger.info("Start profiler")
            self.profiler.start()
        for self.epoch_idx in range(self.start_epoch, self.total_epochs):
            try:
                self.train_one_epoch(self.epoch_idx)
            except Exception as e:
                logger.exception(e)
                if not self.cfg.horovod:
                    dist.barrier()
                    dist.destroy_process_group()
                raise e

        if self.use_profile:
            logger.info("Stopping profiler, do not exit")
            self.profiler.stop()
            logger.info("cpu_time_total")
            print(self.profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
            logger.info("cuda_time_total")
            print(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        if not self.cfg.horovod:
            dist.barrier()
            dist.destroy_process_group()
        logger.info("Training process done")

    def __str__(self) -> str:
        table = PrettyTable()
        table.field_names = ["System Info", "Basic Info", "Loss", "Time Cost"]
        idx = torch.cuda.current_device()
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            idx = int(ids[idx])
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        gpuUtilRate = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory = psutil.virtual_memory()
        unit = 1024 * 1024 * 1024
        sys_info = [
            pynvml.nvmlDeviceGetName(handle),
            f'cpu core: {psutil.cpu_count()}',
            f'gpu mem: {memoryInfo.used / unit:.1f}/{memoryInfo.total / unit:.1f}',
            f'sys mem: {memory.used / unit:.1f}/{memory.total / unit:.1f}',
            f'gpu usage: {gpuUtilRate:.1f}',
            f'cpu usage: {psutil.cpu_percent()}',
        ]

        time_cost_mean = {k: sum(v) / self.summary_step for k, v in self.time_cost.items()}
        one_step_time_cost = sum(time_cost_mean.values())
        time_cost = []
        for key, value in time_cost_mean.items():
            time_cost.append(f"{key}: {value:.3f} ({value/one_step_time_cost*100:.2f}%)")

        time_consumed = timedelta(seconds=int(time.time() - self.time_anchor))
        eta_training_seconds = one_step_time_cost * (self.total_steps - self.global_step)
        eat_training = timedelta(seconds=int(eta_training_seconds))
        lr_current = max([round(x['lr'], 5) for x in self.optimizer.param_groups])

        dataset_size = ','.join([str(x) for x in self.dataset_size])
        batch_size = ','.join([str(x) for x in self.batch_per_gpu])
        basic_info = [
            f"task_name: {self.task_name}",
            f"num_gpus: {self.num_gpus}",
            f"batch_size: {batch_size}",
            f"dataset: {dataset_size}",
            f"lr_per_gpu: {self.lr_per_gpu}",
            f"lr_total: {self.lr_total}",
            f"lr_current: {lr_current}",
            f"epoch: {self.epoch_idx}/{self.total_epochs}",
            f"step: {self.step_idx + 1}/{self.steps_per_epoch}",
            f"global_step: {self.global_step}",
        ]
        loss = [f"loss: {self.loss_vars['loss']:.3f}"]
        loss.extend([f"{loss_name}: {loss_value:.3f}"
                     for loss_name, loss_value in self.loss_vars.items()
                     if loss_name != "loss"])

        if self.rank == 0:
            self.writer.add_scalar("lr", lr_current, self.global_step, new_style=True)
            for loss_name, loss_value in self.loss_vars.items():
                self.writer.add_scalar(f"loss/{loss_name}", loss_value, self.global_step, new_style=True)
            if self.cfg.wandb.enabled:
                info = {"LR/lr": lr_current, "Loss/train": self.loss_vars["loss"]}
                info.update({f"Loss/train/{k}": v for k, v in self.loss_vars.items() if k != "loss"})
                wandb.log(info, step=self.global_step, commit=True)

        one_step_samples = sum(self.batch_per_gpu) * self.num_gpus
        if one_step_time_cost > 0.0:
            time_cost = [
                f"sample/s: {one_step_samples / one_step_time_cost:.3f}",
            ] + time_cost

        time_cost = [
            f"time_consumed: {time_consumed}",
            f"eta_training: {eat_training}",
        ] + time_cost

        for data_row_list in itertools.zip_longest(sys_info, basic_info, loss, time_cost, fillvalue=""):
            table.add_row(data_row_list)
        return f"logdir: {self.work_dir}\n{table.get_string()}"
