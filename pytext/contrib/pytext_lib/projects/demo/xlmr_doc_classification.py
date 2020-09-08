#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json

from omegaconf import OmegaConf
from pytext.contrib.pytext_lib.tasks.fb_xlmr_doc_task import (
    XlmrForDocClassificationTask,
)


config_json = """
{
  "config": {
    "data": {
      "train_path": "manifold://nlp_technologies/tree/pytext/public/datasets/glue_sst2/dev_100.tsv",
      "val_path": "manifold://nlp_technologies/tree/pytext/public/datasets/glue_sst2/dev_100.tsv",
      "test_path": "manifold://nlp_technologies/tree/pytext/public/datasets/glue_sst2/dev_100.tsv",
      "columns": [
        "text",
        "label"
      ],
      "batch_size": 32
    },
    "transform": {
      "label_names": [
        "0",
        "1"
      ],
      "vocab_path": "manifold://nlp_technologies/tree/pytext/public/vocabs/vocab_dummy.txt"
    },
    "model": {
      "model_path": "manifold://nlp_technologies/tree/pytext/public/models/xlmr/xlmr_dummy.pt",
      "dense_dim": 0,
      "embedding_dim": 32,
      "out_dim": 2,
      "vocab_size": 100,
      "num_attention_heads": 1,
      "num_encoder_layers": 1,
      "output_dropout": 0.4,
      "bias": true
    },
    "optim": {
      "lr": 0.00001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 0,
      "amsgrad": false
    },
    "trainer": {
      "max_epochs": 2,
      "gpus": null,
      "distributed_backend": null,
      "num_nodes": 1,
      "checkpoint_callback": false,
      "default_root_dir": null,
      "replace_sampler_ddp": false,
      "num_sanity_val_steps": 1,
      "log_gpu_memory": null,
      "reload_dataloaders_every_epoch": false,
      "row_log_interval": 1,
      "weights_summary": null
    }
  }
}
"""
config_dict = json.loads(config_json)
cfg = OmegaConf.create(config_dict).config

task = XlmrForDocClassificationTask(cfg)
