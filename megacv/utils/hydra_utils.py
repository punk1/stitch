#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-06-03 08:49:18
"""

from pathlib import Path
from typing import List

import hydra
import yaml
from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.global_hydra import GlobalHydra
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import OmegaConf

from .dict_utils import Dict


class ExpSearch(SearchPathPlugin):

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="workspace", path="/workspace/megacv/configs")
        parents = list(self.config.parents)
        if len(parents) >= 3:
            search_path.append(provider=parents[-3].name, path=str(parents[-3].absolute()))
            dirs = [x for x in parents[-3].rglob('*') if x.is_dir()]
            for path in dirs:
                search_path.append(provider=path.name, path=str(path.absolute()))


def load_cfg(config: str, overrides: List = []) -> Dict:
    config = Path(config)
    if not GlobalHydra().is_initialized():
        ExpSearch.config = config
        Plugins.instance().register(ExpSearch)
        hydra.initialize_config_dir(str(list(config.parents)[-2].absolute()), version_base=None)
        OmegaConf.register_new_resolver("eval", eval)

    cfg = hydra.compose(config_name=config.name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return Dict(yaml.safe_load(OmegaConf.to_yaml(cfg)))
