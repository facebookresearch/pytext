#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Data:
    train_path: str
    val_path: str
    test_path: str
    field_names: List[str]
    batch_size: int
    length: Optional[int]
    is_shuffle: bool = True
    is_cycle: bool = False


@dataclass
class Transform:
    doc_label_vocab: List[str]
    slot_label_vocab: List[str]
    vocab_path: str
    max_seq_len: Optional[int] = 256


@dataclass
class Model:
    use_intent: bool
    loss_doc_weight: float
    pretrain_embed: Optional[str]
    embed_dim: int
    slot_kernel_num: int
    slot_kernel_sizes: List[int]
    doc_kernel_num: int
    doc_kernel_sizes: List[int]
    dropout: float
    slot_decoder_hidden_dims: Optional[List[int]]
    doc_decoder_hidden_dims: Optional[List[int]]
    slot_bias: bool = True
    doc_bias: bool = True


@dataclass
class Optimizer:
    lr: float
    betas: List[float]
    eps: float = 0.00000001
    weight_decay: float = 0
    amsgrad: bool = False


@dataclass
class TrainerWrapper:
    max_epochs: int
    max_steps: Optional[int]


@dataclass
class TaskWrapper:
    data: Data
    transforms: Transform
    model: Model
    optimizer: Optimizer


@dataclass
class IntentSlotConfigWrapper:
    task: TaskWrapper
    trainer: TrainerWrapper
