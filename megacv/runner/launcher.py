#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-12 13:19:40
"""

import datetime
import json
import logging
import os
import resource
import socket
import warnings
from shutil import copytree, ignore_patterns
from typing import Any, Dict

import horovod.torch as hvd
import numpy as np
import onnx
import torch
import torch.distributed as dist
import wandb
import yaml
from mmengine.analysis import ActivationAnalyzer, FlopAnalyzer
from mmengine.analysis.print_helper import complexity_stats_table
from mpi4py import MPI
from onnxsim import simplify

from ..models.builder import MODELS
from ..utils import (CheckpointManager, EMAModel, FileManager, get_dist_info,
                     log_init)
from .builder import INFERER, QUANTIZER, TRAINER

warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)
# https://github.com/pytorch/pytorch/issues/973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
base_soft_limit = rlimit[0]
hard_limit = rlimit[1]
soft_limit = min(max(655350, base_soft_limit), hard_limit)
resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def check_connection(ip: str, port: int) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return s.connect_ex((ip, port)) == 0
    except:
        return False
    finally:
        s.close()


def select_port(port: int = 12345) -> int:
    while check_connection('127.0.0.1', port):
        port += 1
    return port


def horovod_init(cfg: Dict[str, Any], group: str = 'train') -> None:
    hvd.init()
    rank = hvd.rank()
    size = hvd.size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    if rank == 0 and cfg.wandb.enabled:
        wandb.login()
        host_name = os.environ.get('HOSTNAME', '')
        exp_name = '-'.join(host_name.split('-')[:-3]) if host_name else cfg.task_name
        exp_name = exp_name[:-5] if exp_name.endswith('-eval') else exp_name
        name = datetime.datetime.now().strftime("%m%d-%H%M")
        cfg.wandb.track.gpus = size
        wandb.init(
            config=cfg.wandb.track,
            project=cfg.wandb.project,
            group=group,
            job_type=exp_name,
            name=name,
        )


def mpi_init(cfg: Dict[str, Any], group: str = 'train') -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if os.environ.get('WORKERHOSTS'):
        nodes = json.loads(os.environ['WORKERHOSTS'])
        rank = int(os.environ['JOBINDEX']) * size + rank
        size = size * len(nodes)
        addr = nodes[0].split('.')[0]
        port = nodes[0].split(':')[1]
    elif os.environ.get('MASTER_ADDR'):
        addr = os.environ['MASTER_ADDR']
        rank = int(os.environ['RANK']) * size + rank
        size = size * int(os.environ['WORLD_SIZE'])
        port = os.environ['MASTER_PORT']
    elif os.environ.get('TF_CONFIG'):
        doc = json.loads(os.environ['TF_CONFIG'])
        rank = doc['task']['index'] * size + rank
        nodes = doc['cluster']['worker']
        size = size * len(nodes)
        addr, port = nodes[0].split(':')
    else:
        addr = '127.0.0.1'
        if rank == 0:
            port = str(select_port())
            for i in range(1, size):
                comm.send(port, dest=i)
        else:
            port = comm.recv()

    init_method = f'tcp://{addr}:{port}'
    dist.init_process_group(backend='nccl',
                            init_method=init_method,
                            rank=rank,
                            world_size=size)
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    if rank == 0 and cfg.wandb.enabled:
        wandb.login()
        host_name = os.environ.get('HOSTNAME', '')
        exp_name = '-'.join(host_name.split('-')[:-3]) if host_name else cfg.task_name
        exp_name = exp_name[:-5] if exp_name.endswith('-eval') else exp_name
        name = datetime.datetime.now().strftime("%m%d-%H%M")
        cfg.wandb.track.gpus = size
        wandb.init(
            config=cfg.wandb.track,
            project=cfg.wandb.project,
            group=group,
            job_type=exp_name,
            name=name,
        )


def train(cfg: Dict[str, Any]) -> None:
    horovod_init(cfg, 'train') if cfg.horovod else mpi_init(cfg, 'train')
    file_manager = FileManager(cfg)
    log_init(os.path.join(file_manager.log_dir, "train.log"))

    rank, _ = get_dist_info()
    if rank == 0:
        logger = logging.getLogger()
        logger.info("config:\n" + json.dumps(cfg, indent=4, ensure_ascii=False))
        for dirname in ['configs', 'megacv', 'modules']:
            if os.path.exists(dirname) and not os.environ.get('WORKERHOSTS'):
                copytree(dirname, os.path.join(file_manager.code_dir, dirname),
                         ignore=ignore_patterns('__pycache__'),
                         dirs_exist_ok=True)

        cfg_file = os.path.join(file_manager.code_dir, "task.yaml")
        yaml.dump(cfg.dict(), open(cfg_file, "w"), default_flow_style=False, sort_keys=False)

    trainer = TRAINER.build(cfg.trainer, default_args={"cfg": cfg})
    trainer.train()


def infer(cfg: Dict[str, Any]) -> None:
    horovod_init(cfg, 'eval') if cfg.horovod else mpi_init(cfg, 'eval')
    file_manager = FileManager(cfg)
    log_init(os.path.join(file_manager.log_dir, "eval.log"))
    if cfg.monitor or (not cfg.resume and cfg.pretrained is None):
        cfg.resume = True
    logger = logging.getLogger()
    logger.info("config:\n" + json.dumps(cfg, indent=4, ensure_ascii=False))

    inferer = INFERER.build(cfg.inferer, default_args={"cfg": cfg})
    if cfg.monitor:
        inferer.infer_infinite()
    else:
        inferer.infer()


def quant(cfg: Dict[str, Any]) -> None:
    log_init()
    quantizer = QUANTIZER.build(cfg.quantizer, default_args={"cfg": cfg})
    quantizer.quant()


def export(cfg: Dict[str, Any]) -> None:
    logger = log_init()
    file_manager = FileManager(cfg)
    logger.info(f"export to {cfg.export.name}")
    inputs = {"inputs": {k: torch.zeros(v) for k, v in cfg.export.inputs.items()}}
    model = MODELS.build(cfg.model).eval()
    ema_model = EMAModel(model, **cfg.ema)
    ckpt_manager = CheckpointManager(
        save_dir=file_manager.ckpt_dir,
        model=ema_model,
        resume=cfg.resume,
        pretrained=cfg.pretrained,
    )
    ckpt_manager.load_model()
    torch.onnx.export(
        model,
        inputs,
        cfg.export.name,
        verbose=cfg.export.verbose,
        opset_version=cfg.export.opset_version,
        input_names=cfg.export.input_names,
        output_names=cfg.export.output_names)
    model = onnx.load(cfg.export.name)
    model_sim, check = simplify(model)
    if check:
        logger.info("onnx simplify succeed")
        onnx.save(model_sim, cfg.export.name)
    else:
        logger.info("onnx simplify failed")


def flops(cfg: Dict[str, Any]) -> None:
    model = MODELS.build(cfg.model).eval()
    inputs = {k: torch.randn(v) for k, v in cfg.export.inputs.items()}
    flop_handler = FlopAnalyzer(model, inputs)
    activation_handler = ActivationAnalyzer(model, inputs)
    complexity_table = complexity_stats_table(
        flops=flop_handler,
        activations=activation_handler,
        show_param_shapes=True,
    )
    print(complexity_table)


def fusion(cfg: Dict[str, Any]) -> None:
    print(json.dumps(cfg, indent=4, ensure_ascii=False))
    logger = log_init()

    key = 'ema_state_dict' if cfg.ema.enabled else 'model_state_dict'
    base_ckpt = {}
    update_ckpt = {}
    base_ckpt_file = cfg.ckpt or cfg.pretrained
    assert os.path.exists(base_ckpt_file), f'{base_ckpt_file} not exists'
    if base_ckpt_file:
        base_ckpt = CheckpointManager.load_ckpt(base_ckpt_file)
        base_ckpt = base_ckpt.get('model', base_ckpt)
        base_ckpt = base_ckpt.get(key, base_ckpt)
    if cfg.ckpts:
        cfg.ckpts = [cfg.ckpts] if isinstance(cfg.ckpts, str) else cfg.ckpts
        for ckpt_path in cfg.ckpts:
            ckpt = CheckpointManager.load_ckpt(ckpt_path)
            ckpt = ckpt.get('model', ckpt)
            ckpt = ckpt.get(key, ckpt)
            for k, v in ckpt.items():
                if k in update_ckpt:
                    assert (update_ckpt[k] == v).all(), f"{k} in {ckpt_path} conflict"
                update_ckpt[k] = v
    logger.info(f"base params: {len(base_ckpt)}")
    base_ckpt.update(update_ckpt)
    logger.info(f"fusion params: {len(base_ckpt)}")
    torch.save(base_ckpt, f"{cfg.task_name}.pth")
