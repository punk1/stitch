#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2023-04-06 11:06:24
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict

import onnx
import torch
from aimet_common.defs import QuantScheme
from aimet_common.utils import AimetLogger
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.quantsim import QuantizationSimModel
from onnxsim import simplify
from tqdm import tqdm

from ..data.builder import DATALOADER, DATASETS
from ..data.dataloader import AimetDataLoader
from ..models.builder import MODELS
from ..utils import CheckpointManager, FileManager
from .builder import QUANTIZER

AimetLogger.set_level_for_all_areas(logging.ERROR)
logger = logging.getLogger()


@QUANTIZER.register_module()
class Quantizer:

    """MegaCV default quantizer using aimet
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        equalize: bool = False,
        num_batches: int = 10,
        output: str = 'models',
        prefix: str = 'aimet',
    ):
        self.cfg = cfg
        self.equalize = equalize
        self.num_batches = num_batches
        self.output = Path(output)
        self.prefix = prefix
        self.output.mkdir(parents=True, exist_ok=True)
        self.file_manager = FileManager(cfg)
        self.build_components()

    def build_dataloader(self, cfg):
        dataset = DATASETS.build(cfg.dataset, default_args={"mode": "train"})
        default_args = {
            "dataset": dataset,
            "shuffle": True,
            "seed": 0,
            "start_epoch": 0,
            "total_epochs": math.inf,
        }
        return DATALOADER.build(cfg.dataloader, default_args=default_args)

    def build_components(self):
        self.dataloader = {}
        if self.cfg.task_configs:
            if isinstance(self.cfg.batch_size, int):
                self.cfg.batch_size = [self.cfg.batch_size] * len(self.cfg.task_configs)
            for (task_name, cfg), batch_size in zip(self.cfg.task_configs.items(), self.cfg.batch_size):
                cfg.dataloader.batch_size = batch_size
                self.dataloader[task_name] = self.build_dataloader(cfg)
        else:
            self.dataloader[self.cfg.task_name] = self.build_dataloader(self.cfg)

        self.dataloader = AimetDataLoader(list(self.dataloader.values()), self.num_batches)
        self.model = MODELS.build(self.cfg.model).cuda()
        self.ckpt_manager = CheckpointManager(
            model=self.model,
            save_dir=self.file_manager.ckpt_dir,
            resume=self.cfg.resume,
            pretrained=self.cfg.pretrained,
            to_cuda=True,
        )
        self.ckpt_manager.load_model()

        if self.equalize:
            equalize_model(self.model, tuple(self.cfg.export.inputs.img))

    @staticmethod
    def compute_forward(model, dataloader):
        model.eval()
        with torch.no_grad():
            for inputs in tqdm(dataloader):
                model(inputs)
        model.train()

    def quant(self):
        dummy_input = {k: torch.randn(v).cuda() for k, v in self.cfg.export.inputs.items()}
        quantsim = QuantizationSimModel(model=self.model,
                                        quant_scheme=QuantScheme.training_range_learning_with_tf_init,
                                        dummy_input=dummy_input,
                                        rounding_mode='nearest',
                                        default_output_bw=8,
                                        default_param_bw=8,
                                        in_place=False)

        logger.info("compute encoding start")
        quantsim.compute_encodings(forward_pass_callback=self.compute_forward,
                                   forward_pass_callback_args=self.dataloader)
        logger.info("compute encoding succeed")

        dummy_input = {"inputs": {k: torch.randn(v) for k, v in self.cfg.export.inputs.items()}}
        logger.info("export start")
        quantsim.export(
            path=self.output,
            filename_prefix=self.prefix,
            dummy_input=dummy_input,
            onnx_export_args=OnnxExportApiArgs(
                opset_version=self.cfg.export.opset_version,
                input_names=self.cfg.export.input_names,
                output_names=self.cfg.export.output_names),
        )
        filename = str(self.output / f'{self.prefix}.onnx')
        logger.info(f"export succeed: {filename}")

        model = onnx.load(filename)
        model_sim, check = simplify(model)
        if check:
            logger.info("onnx simplify succeed")
            onnx.save(model_sim, filename)
        else:
            logger.info("onnx simplify failed")
