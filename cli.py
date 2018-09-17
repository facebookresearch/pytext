#!/usr/bin/env python3

from pytext.config import PyTextConfig
from pytext.fb.utils.data_utils import fetch_embedding_and_update_config
from pytext.workflow import test_model, train_model

from .args import parse_config


def run_trainer():
    config = parse_config()
    if config.test_given_snapshot:
        test_model(config)
    else:
        config = fetch_embedding_and_update_config(config)
        print(
            "Using pretrained embeddings from {}".format(
                config.jobspec.features.word_feat.pretrained_embeddings_path
            )
        )
        print("Starting training...")
        train_model(config)
        print("Starting testing...")
        test_config_dict = config._asdict()
        test_config_dict["load_snapshot_path"] = config.save_snapshot_path
        test_config = PyTextConfig(**test_config_dict)
        test_model(test_config)
