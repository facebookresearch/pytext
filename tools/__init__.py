#!/usr/bin/env python3

import sys
import logging


def init_logger(logger_name="", log_file_path=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    log_write_target = log_file_path if log_file_path else sys.stdout
    handler = logging.StreamHandler(log_write_target)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
