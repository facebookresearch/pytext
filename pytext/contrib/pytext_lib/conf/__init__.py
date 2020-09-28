#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# @manual "//github/facebookresearch/hydra:hydra"
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from stl.lightning.conf.trainer import TrainerConf


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
    _target_: str = "pytext.contrib.pytext_lib.transforms.fb_doc_transform.DocTransform"
    label_names: Optional[List[str]] = field(default_factory=lambda: ["True", "False"])
    vocab_path: Optional[str] = None


@dataclass
class ModelConf:
    pass


@dataclass
class RobertaModelConf(ModelConf):
    model_path: Optional[str] = None
    dense_dim: Optional[int] = 0
    embedding_dim: int = 32
    out_dim: int = 2
    vocab_size: int = 100
    num_attention_heads: int = 1
    num_encoder_layers: int = 1
    output_dropout: float = 0.4
    bias: bool = True
    # TODO: @steven make Roberta model accept primitive atgs, then add _target_


@dataclass
class DocClassificationModelConf(ModelConf):
    _target_: str = "pytext.contrib.pytext_lib.models.DocClassificationModel"
    # word embedding config
    pretrained_embeddings_path: str = MISSING
    embedding_dim: int = MISSING
    mlp_layer_dims: Optional[List[int]] = field(default_factory=list)
    lowercase_tokens: bool = False
    skip_header: bool = True
    delimiter: str = " "
    # DocNN config
    kernel_num: int = 100
    kernel_sizes: Optional[List[int]] = field(default_factory=list)
    pooling_type: str = "max"
    dropout: float = 0.4
    # decoder config
    dense_dim: int = 0
    decoder_hidden_dims: Optional[List[int]] = field(default_factory=list)
    out_dim: int = 2


@dataclass
class OptimConf:
    _target_: str = "torch.optim.AdamW"
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False


@dataclass
class ClassificationMetricReporterConf:
    recall_at_precision_thresholds: List[float] = field(
        default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 0.9]
    )


# TODO: @stevenliu merge to TaskConf
@dataclass
class DocClassificationConfig:
    # MISSING doesn't work with non "str" type in flow
    data: DataConf
    transform: TransformConf
    model: RobertaModelConf
    optim: OptimConf
    metric_reporter: ClassificationMetricReporterConf = ClassificationMetricReporterConf()
    trainer: TrainerConf = TrainerConf()
    defaults: List[Any] = field(default_factory=lambda: project_defaults)


@dataclass
class DataModuleConf:
    pass


@dataclass
class DocClassificationDataModuleConf(DataModuleConf):
    _target_: str = "pytext.contrib.pytext_lib.data.datamodules.doc_classification.DocClassificationDataModule"
    train_path: str = MISSING
    val_path: str = MISSING
    test_path: str = MISSING
    columns: List[str] = field(default_factory=lambda: ["text", "label"])
    column_mapping: Optional[Dict[str, str]] = None
    delimiter: str = "\t"
    batch_size: Optional[int] = None
    is_shuffle: bool = True
    chunk_size: int = 1000
    is_cycle: bool = False
    length: Optional[int] = None
    transform: TransformConf = TransformConf()


@dataclass
class TaskConf:
    _target_: str = "pytext.contrib.pytext_lib.tasks.fb_doc_classification_task.DocClassificationTask"
    model: ModelConf = MISSING
    optim: OptimConf = MISSING


@dataclass
class PyTextConf:
    datamodule: DataModuleConf = MISSING
    task: TaskConf = TaskConf()
    trainer: TrainerConf = TrainerConf()
    defaults: List[Any] = field(
        default_factory=lambda: (
            {"datamodule": "doc_classification"},
            {"task/model": "doc_model_with_spm"},
            {"task/optim": "fairseq_adam"},
            {"trainer": "cpu"},
        )
    )


cs = ConfigStore.instance()
cs.store(group="data", name="sst2", node=DataConf)
cs.store(group="data", name="sst2_dummy", node=DataConf)
cs.store(group="transform", name="roberta", node=TransformConf)

cs.store(group="schema/task/model", name="xlmr", node=RobertaModelConf)
cs.store(group="task/model", name="xlmr_base", node=RobertaModelConf)
cs.store(group="task/model", name="xlmr_dummy", node=RobertaModelConf)

cs.store(group="schema/task/model", name="doc_model", node=DocClassificationModelConf)
cs.store(group="task/model", name="doc_model_dummy", node=DocClassificationModelConf)
cs.store(group="task/model", name="doc_model_with_spm", node=DocClassificationModelConf)
cs.store(group="task/model", name="doc_model_with_xlu", node=DocClassificationModelConf)

cs.store(group="task/optim", name="adamw", node=OptimConf)
cs.store(group="task/optim", name="fairseq_adam", node=OptimConf)

# TODO: @stevenliu merge it to task_config
cs.store(name="config", node=DocClassificationConfig)

cs.store(
    group="schema/datamodule",
    name="doc_classification",
    node=DocClassificationDataModuleConf,
)
cs.store(
    group="datamodule", name="doc_classification", node=DocClassificationDataModuleConf
)
cs.store(
    group="datamodule",
    name="doc_classification_dummy",
    node=DocClassificationDataModuleConf,
)
cs.store(name="pytext_config", node=PyTextConf)
