#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""log_manager.

Author:     kaizhang
Created at: 2022-04-01 19:00:46
"""

import logging
import logging.handlers

from coloredlogs import BasicFormatter, ColoredFormatter

from .dist_utils import get_dist_info, master_only

LOGGING_FORMAT = "[%(levelname)s %(asctime)s %(module)s:%(lineno)d] %(message)s"


@master_only
def log_init(filename: str = None, name: str = None, level: str = "INFO") -> None:
    """Init logger with stream handler for specific name."""
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    logger.propagate = False

    logger.handlers = [x for x in logger.handlers if not isinstance(x, logging.StreamHandler)]
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(LOGGING_FORMAT))
    logger.addHandler(handler)

    if filename is not None:
        handler = logging.handlers.WatchedFileHandler(filename=filename, mode='a', encoding='utf-8')
        handler.setFormatter(BasicFormatter(LOGGING_FORMAT))
        logger.addHandler(handler)

    return logger


def log_reset(logger: logging.Logger, level: str = "INFO"):
    rank, _ = get_dist_info()
    if rank != 0:
        logger.handlers = []
        return logger

    logger.setLevel(level.upper())
    for handler in logger.handlers:
        handler.setLevel(level.upper())
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(ColoredFormatter(LOGGING_FORMAT))
        elif isinstance(handler, logging.handlers.WatchedFileHandler):
            handler.setFormatter(BasicFormatter(LOGGING_FORMAT))

    if not any([isinstance(x, logging.StreamHandler) for x in logger.handlers]):
        handler = logging.StreamHandler()
        handler.setFormatter(ColoredFormatter(LOGGING_FORMAT))
        logger.addHandler(handler)

    return logger
