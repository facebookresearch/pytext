#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

# @manual "//github/facebookresearch/hydra:hydra"
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


project_defaults = [
    {"data": "sst2"},
    {"transform": "roberta"},
    {"model": "xlmr_base"},
    {"optim": "fairseq_adam"},
    {"trainer": "cpu"},
]


@dataclass
class DataConf:
    # MISSING doesn't work with non "str" type in flow
    train_path: str = MISSING
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    columns: List[str] = field(default_factory=lambda: ["text", "label"])
    batch_size: int = 8


@dataclass
class TransformConf:
    label_names: Optional[List[str]] = field(default_factory=lambda: ["True", "False"])
    vocab_path: str = MISSING


@dataclass
class ModelConf:
    model_path: Optional[str] = None
    dense_dim: Optional[int] = 0
    embedding_dim: int = 32
    out_dim: int = 2
    vocab_size: int = 100
    num_attention_heads: int = 1
    num_encoder_layers: int = 1
    output_dropout: float = 0.4
    bias: bool = True


@dataclass
class OptimConf:
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False


@dataclass
class TrainerConf:
    """Config for Lightning Trainer
    See the usage of each arg in pytorch_lightning/trainer/trainer.py
    """

    max_epochs: int = 1
    # `gpus` should be of type Optional[Union[List[int], str, int]]
    # However, Hydra doesn't support Union yet, and flow type doesn't support Any
    gpus: Optional[int] = None
    distributed_backend: Optional[str] = None
    num_nodes: int = 1
    checkpoint_callback: bool = False
    default_root_dir: Optional[str] = None
    replace_sampler_ddp: bool = False
    num_sanity_val_steps: int = 1
    log_gpu_memory: Optional[str] = None
    reload_dataloaders_every_epoch: bool = False
    row_log_interval: int = 10
    weights_summary: Optional[str] = "full"


@dataclass
class DocClassificationConfig:
    # MISSING doesn't work with non "str" type in flow
    data: DataConf
    transform: TransformConf
    model: ModelConf
    optim: OptimConf
    trainer: TrainerConf = TrainerConf()
    defaults: List[Any] = field(default_factory=lambda: project_defaults)


cs = ConfigStore.instance()
cs.store(group="data", name="sst2", node=DataConf)
cs.store(group="data", name="sst2_dummy", node=DataConf)
cs.store(group="transform", name="roberta", node=TransformConf)
cs.store(group="model", name="xlmr", node=ModelConf)
cs.store(group="model", name="xlmr_dummy", node=ModelConf)
cs.store(group="optim", name="fairseq_adam", node=OptimConf)
cs.store(group="trainer", name="cpu", node=TrainerConf)
cs.store(group="trainer", name="single_gpu", node=TrainerConf)
cs.store(group="trainer", name="multi_gpu", node=TrainerConf)

cs.store(name="config", node=DocClassificationConfig)
