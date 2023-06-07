#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author:   zhangkai
Created:  2022-04-24 17:19:12
"""

import os
import sys
from importlib import import_module
from pathlib import Path
from shutil import copy, copytree


def help():
    print("""Usage: megacv ACTION CONFIG [OPTIONS]

Args:
    action: train, infer, export, flops, quant
    config: config file path

Options:
    config overwrite in command line
""")


def init():
    modules = [
        'modules/data/datasets',
        'modules/data/transforms',
        'modules/data/visualizer',
        'modules/evaluators',
        'modules/models/detectors',
        'modules/models/backbones',
        'modules/models/necks',
        'modules/models/heads',
        'modules/models/layers',
        'modules/models/utils',
        'modules/models/postprocessing',
        'modules/runner',
    ]
    for module in modules:
        file = Path(module) / '__init__.py'
        file.parent.mkdir(parents=True, exist_ok=True)
        if not file.exists():
            file.write_text('# -*- coding: utf-8 -*-\n\n')

    file = Path('modules/__init__.py')
    if not file.exists():
        file.write_text('''# -*- coding: utf-8 -*-
from . import data, evaluators, models, runner  # noqa
''')
    file = Path('modules/data/__init__.py')
    if not file.exists():
        file.write_text('''# -*- coding: utf-8 -*-
from . import datasets, transforms, visualizer  # noqa
''')
    file = Path('modules/models/__init__.py')
    if not file.exists():
        file.write_text('''# -*- coding: utf-8 -*-
from . import detectors, backbones, necks, heads, layers, utils, postprocessing  # noqa
''')

    Path('configs').mkdir(parents=True, exist_ok=True)
    copy('/workspace/megacv/configs/template_adas.yaml', 'configs/template_adas.yaml')
    copy('/workspace/megacv/configs/template.yaml', 'configs/template.yaml')
    copy('/workspace/megacv/configs/pld.yaml', 'configs/pld.yaml')
    copy('/workspace/megacv/.gitignore', '.')
    copy('/workspace/megacv/.pre-commit-config.yaml', '.')

    Path('scripts').mkdir(parents=True, exist_ok=True)
    copytree('/workspace/megacv/scripts', 'scripts', dirs_exist_ok=True)


def main():
    sys.path.insert(0, os.getcwd())
    for module in ['megacv', 'modules']:
        if os.path.exists(module):
            sys.modules.pop(module, None)
            import_module(module)

    from megacv.runner import launcher
    from megacv.utils import load_cfg

    action = sys.argv[1].replace('-', '_')
    overrides = [x.lstrip('-') for x in sys.argv[3:]]
    cfg = load_cfg(sys.argv[2], overrides)
    if cfg.task_configs:
        cfg.task_configs = {k: load_cfg(v) for k, v in cfg.task_configs.items()}
    getattr(launcher, action)(cfg)


def run():
    if len(sys.argv) >= 2 and sys.argv[1] == 'init':
        init()
    elif len(sys.argv) < 3:
        help()
    else:
        main()


if __name__ == '__main__':
    run()
