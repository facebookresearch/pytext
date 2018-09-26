#!/usr/bin/env python3

from pytext.config import PyTextConfig
from pytext.workflow import test_model, train_model

from .args import parse_config


def run_job():
    config = parse_config()
    if config.test_given_snapshot:
        test_model(config)
    else:
        print("Starting training...")
        train_model(config)
        print("Starting testing...")
        test_config_dict = config._asdict()
        test_config_dict["load_snapshot_path"] = config.save_snapshot_path
        test_config = PyTextConfig(**test_config_dict)
        test_model(test_config)
