#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-17 19:09:17
"""

import os
from typing import Any, Dict

from .dist_utils import get_dist_info


class FileManager:

    def __init__(self, cfg: Dict[str, Any]):
        self._work_dir = os.path.join(cfg.work_dir, cfg.task_name)
        self._ckpt_dir = os.path.join(self._work_dir, "ckpt")
        self._code_dir = os.path.join(self._work_dir, "code")
        self._log_dir = os.path.join(self._work_dir, "log")
        self._summary_dir = os.path.join(self._work_dir, "summary")
        self._eval_dir = os.path.join(self._work_dir, "eval")
        rank, _ = get_dist_info()
        if rank == 0:
            os.makedirs(self._ckpt_dir, exist_ok=True)
            os.makedirs(self._code_dir, exist_ok=True)
            os.makedirs(self._log_dir, exist_ok=True)
            os.makedirs(self._summary_dir, exist_ok=True)
            os.makedirs(self._eval_dir, exist_ok=True)

    @property
    def work_dir(self):
        return self._work_dir

    @property
    def ckpt_dir(self):
        return self._ckpt_dir

    @property
    def code_dir(self):
        return self._code_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def summary_dir(self):
        return self._summary_dir

    @property
    def eval_dir(self):
        return self._eval_dir
