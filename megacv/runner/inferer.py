#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-12 19:28:05
"""

import collections
import inspect
import itertools
import logging
import os
import random
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import horovod.torch as hvd
import numpy as np
import psutil
import pynvml
import torch
import torch.distributed as dist
import wandb
from prettytable import PrettyTable

from ..data.builder import DATALOADER, DATASETS
from ..evaluators.builder import EVALUATORS
from ..models.builder import MODELS
from ..parallel import DistributedDataParallel
from ..utils import (CheckpointManager, EMAModel, FileManager, Stopwatch,
                     get_dist_info, log_reset)
from .builder import INFERER

logger = logging.getLogger()


@INFERER.register_module()
class Inferer:

    """MegaCV default inferer

    Args:
        cfg (dict): complete parameters in yaml
        interval (int): seconds to check new checkpoint, Default 3600
        summary_step (int): logging frequency, Default: 20
        use_fp16 (bool): whether to use fp16 training, Default: False
        use_syncbn (bool): whether to use sync batchnorm, Default: False
        use_deterministic (bool): whether to enable deterministic algorithms, Default: False
        seed (int): random seed, Default: 1234
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        interval: int = 3600,
        summary_step: int = 50,
        use_fp16: bool = False,
        use_syncbn: bool = False,
        global_syncbn: bool = False,
        use_deterministic: bool = False,
        use_benchmark: bool = True,
        switch_to_deploy: bool = False,
        seed: int = 1234,
        **kwargs,
    ):
        torch.cuda.empty_cache()
        pynvml.nvmlInit()

        self.cfg = cfg
        self.task_name = cfg.task_name
        self.interval = interval
        self.summary_step = summary_step
        self.use_fp16 = use_fp16
        self.use_syncbn = use_syncbn
        self.global_syncbn = global_syncbn
        self.use_deterministic = use_deterministic
        self.use_benchmark = use_benchmark
        self.switch_to_deploy = switch_to_deploy
        self.seed = seed

        assert not (cfg.monitor and switch_to_deploy), "monitor and switch_to_deploy cannot both be true"

        self.file_manager = FileManager(cfg)
        self.work_dir = Path(self.file_manager.work_dir).absolute()
        self.current_device = torch.cuda.current_device()
        self.rank, self.num_gpus = get_dist_info()
        self.stopwatch = Stopwatch(False)

        self.enable_deterministic()
        self.build_components()

    def enable_deterministic(self) -> None:
        torch.use_deterministic_algorithms(self.use_deterministic)
        torch.backends.cudnn.benchmark = self.use_benchmark
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def enable_syncbn(self) -> None:
        if self.cfg.horovod or self.global_syncbn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        else:
            group_size = min(8, self.num_gpus)
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
                find_unused_parameters=True,
            )
        else:
            hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)

    def build_model(self) -> None:
        self.model = MODELS.build(self.cfg.model)
        self.model = self.model.cuda(device=self.current_device)
        self.model_inputs = inspect.signature(self.model.forward).parameters
        if self.cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if self.use_syncbn:
            self.enable_syncbn()

        if self.num_gpus > 1:
            self.enable_dist()

        self.ema_model = EMAModel(self.model, **self.cfg.ema)

    def build_dataloader(self, cfg):
        dataset = DATASETS.build(cfg.dataset, default_args={"mode": "val"})
        default_args = {"dataset": dataset, "seed": self.seed}
        cfg.dataloader.drop_last = False
        return DATALOADER.build(cfg.dataloader, default_args=default_args)

    def build_components(self) -> None:
        self.dataloader = {}
        self.evaluator = {}
        if self.cfg.task_configs:
            for task_name, cfg in self.cfg.task_configs.items():
                self.dataloader[task_name] = self.build_dataloader(cfg)
                save_dir = os.path.join(self.file_manager.eval_dir, task_name)
                os.makedirs(save_dir, exist_ok=True)
                default_args = {'save_dir': save_dir}
                self.evaluator[task_name] = EVALUATORS.build(cfg.evaluator, default_args=default_args)
        else:
            self.dataloader[self.cfg.task_name] = self.build_dataloader(self.cfg)
            default_args = {'save_dir': self.file_manager.eval_dir}
            self.evaluator[self.cfg.task_name] = EVALUATORS.build(self.cfg.evaluator, default_args=default_args)
        self.dataset_size = {k: v.dataset_size for k, v in self.dataloader.items()}
        self.batch_per_gpu = [x.batch_size for x in self.dataloader.values()]
        self.steps_per_epoch = max([len(x) for x in self.dataloader.values()])

        self.build_model()
        self.ckpt_manager = CheckpointManager(
            model=self.ema_model,
            save_dir=self.file_manager.ckpt_dir,
            resume=self.cfg.resume,
            pretrained=self.cfg.pretrained,
            to_cuda=True,
        )

    def parse_losses(self, losses: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        loss_vars = {}
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                loss_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                loss_vars[loss_name] = sum(loss.mean() for loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        return loss_vars

    def forward(self, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any]]:
        preds = self.model(inputs) if len(self.model_inputs) == 1 else self.model(**inputs)
        if hasattr(self.model, 'post_process'):
            preds = self.model.post_process(preds, inputs)
            return {}, preds
        else:
            losses, preds = ({}, preds) if isinstance(preds, dict) else preds
            return losses, preds

    def infer_one_batch(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            losses, preds = self.forward(inputs)
        self.time_cost["model_cost"].append(self.stopwatch.toc2())
        loss_vars = self.parse_losses(losses)
        return loss_vars, preds

    def infer_one_step(self) -> Dict[str, Any]:
        self.stopwatch.tic()
        self.loss_vars = {}
        results = {}
        task_num = len(self.dataloader)
        for i, (task_name, dataloader) in enumerate(self.dataloader.items()):
            inputs = dataloader.get_batch()
            if inputs is None:
                continue
            if self.cfg.model.type == 'MultiDetector':
                inputs["task"] = task_name
            self.time_cost["data_cost"].append(self.stopwatch.toc2())
            losses, preds = self.infer_one_batch(inputs)
            if task_num > 1:
                losses = {f"{task_name}_{k}": v for k, v in losses.items()}
            self.loss_vars.update(losses)
            for k, v in losses.items():
                self.total_losses[k].append(v.item())
            self.time_cost["model_cost"].append(self.stopwatch.toc2())
            results[task_name] = self.evaluator[task_name].update(preds, inputs)
            self.time_cost["eval_cost"].append(self.stopwatch.toc2())

        self.loss_vars["loss"] = sum(value for key, value in self.loss_vars.items() if key != "loss")
        return results

    def infer_one_epoch(self) -> Tuple[Dict[str, torch.Tensor]]:
        self.model.eval()
        if self.switch_to_deploy:
            for m in self.models():
                if hasattr(m, 'switch_to_deploy'):
                    m.switch_to_deploy()

        for evaluator in self.evaluator.values():
            evaluator.reset()

        self.total_losses = collections.defaultdict(list)
        results = defaultdict(list)
        for self.step_idx in range(self.steps_per_epoch):
            ret = self.infer_one_step()
            for k, v in ret.items():
                results[k].append(v)
            if (self.step_idx + 1) % self.summary_step == 0:
                logger.info(self)
                self.time_cost = defaultdict(list)

        if (self.step_idx + 1) % self.summary_step != 0:
            logger.info(self)
            self.time_cost = defaultdict(list)

        metrics = {}
        for task_name, rets in results.items():
            rets = self.evaluator[task_name].collate(rets, self.dataset_size[task_name])
            if self.rank != 0:
                continue
            metric = self.evaluator[task_name].evaluate(rets)
            if isinstance(metric, (int, float)):
                metric = {task_name: metric}
            elif isinstance(metric, dict):
                metric = {f'{task_name}/{k}': v for k, v in metric.items()}
            elif isinstance(metric, (list, tuple)):
                metric = {f'{task_name}/{i}': v for i, v in enumerate(metric)}
            if isinstance(metric, dict):
                metrics.update(metric)

        losses = {}
        for k, v in self.total_losses.items():
            losses[k] = round(sum(v) / len(v), 3)
        losses["loss"] = sum(value for key, value in losses.items() if key != "loss")
        return metrics, losses

    def infer(self) -> None:
        log_reset(logger)
        if not self.cfg.resume and self.cfg.pretrained is None:
            return logger.error("resume and pretrained cannot both be None !!!")

        self.ckpt_manager.load_model()
        self.time_anchor = time.time()
        self.time_cost = defaultdict(list)
        with torch.no_grad():
            metrics, losses = self.infer_one_epoch()
            logger.info(f"evaluation metrics: {metrics}")
            logger.info(f"evaluation losses: {losses}")

        if not self.cfg.horovod:
            dist.barrier()
            dist.destroy_process_group()
        logger.info("Infer process done")

    def infer_infinite(self) -> None:
        log_reset(logger)
        if not self.cfg.resume:
            return logger.error("resume must be true !!!")

        self.time_anchor = time.time()
        self.time_cost = defaultdict(list)
        last_time = None
        while True:
            ckpt_file = self.ckpt_manager.get_last_ckpt()
            epoch = self.ckpt_manager.get_ckpt_epoch(ckpt_file.name)
            if not ckpt_file.exists():
                logger.info("waiting for first checkpoint")
                time.sleep(60)
                continue

            if last_time is not None:
                if last_time == ckpt_file.stat().st_mtime:
                    if epoch == self.cfg.trainer.total_epochs - 1 or time.time() - last_time >= self.interval:
                        break
                    else:
                        logger.info(f"waiting for next checkpoint, current epoch: {epoch}")
                        time.sleep(60)
                        continue
                for dataloader in self.dataloader.values():
                    dataloader.reset()

            self.ckpt_manager.reload()
            self.ckpt_manager.load_model()
            last_time = ckpt_file.stat().st_mtime
            with torch.no_grad():
                try:
                    metrics, losses = self.infer_one_epoch()
                except Exception as e:
                    logger.exception(e)
                    if not self.cfg.horovod:
                        dist.barrier()
                        dist.destroy_process_group()
                    raise e

                logger.info(f"evaluation metrics[{epoch}]: {metrics}")
                logger.info(f"evaluation losses[{epoch}]: {losses}")
                if self.cfg.wandb.enabled and self.rank == 0:
                    info = {"Loss/eval": losses["loss"]}
                    info.update({f"Loss/eval/{k}": v for k, v in losses.items() if k != "loss"})
                    info.update({f"Metrics/{k}": v for k, v in metrics.items()})
                    wandb.log(info, step=epoch, commit=True)

        if not self.cfg.horovod:
            dist.barrier()
            dist.destroy_process_group()

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
        eta_training_seconds = one_step_time_cost * (self.steps_per_epoch - self.step_idx)
        eat_training = timedelta(seconds=int(eta_training_seconds))
        ckpt_name = ''
        if self.cfg.resume:
            ckpt_name = self.ckpt_manager.get_last_ckpt().stem
        elif isinstance(self.cfg.pretrained, str):
            ckpt_name = Path(self.cfg.pretrained).stem

        dataset_size = ','.join([str(x) for x in self.dataset_size.values()])
        batch_size = ','.join([str(x) for x in self.batch_per_gpu])
        basic_info = [
            f"task_name: {self.task_name}",
            f"ckpt_name: {ckpt_name}",
            f"num_gpus: {self.num_gpus}",
            f"batch_size: {batch_size}",
            f"dataset: {dataset_size}",
            f"step: {self.step_idx + 1}/{self.steps_per_epoch}",
        ]
        loss = [f"loss: {self.loss_vars['loss']:.3f}"]
        loss.extend([f"{loss_name}: {loss_value:.3f}"
                     for loss_name, loss_value in self.loss_vars.items()
                     if loss_name != "loss"])

        one_step_samples = sum(self.batch_per_gpu) * self.num_gpus
        if one_step_time_cost > 0.0:
            time_cost = [
                f"sample/s: {one_step_samples / one_step_time_cost:.3f}",
            ] + time_cost

        time_cost = [
            f"time_consumed: {time_consumed}",
            f"eta_inference: {eat_training}",
        ] + time_cost

        for data_row_list in itertools.zip_longest(sys_info, basic_info, loss, time_cost, fillvalue=""):
            table.add_row(data_row_list)
        return f"logdir: {self.work_dir}\n{table.get_string()}"
