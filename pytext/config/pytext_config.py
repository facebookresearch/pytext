#!/usr/bin/env python3
from enum import Enum
from typing import Any, List, NamedTupleMeta, Optional, Tuple, Union  # noqa

from pytext.common.python_utils import InheritableNamedTupleMeta


class ConfigBase(metaclass=InheritableNamedTupleMeta):
    _root = True


class PlaceHolder:
    pass


class PyTextConfig(ConfigBase):
    # the actual task union types will be generated in runtime
    jobspec: Union[Any, PlaceHolder]
    use_cuda_if_available: bool = True
    # Total Number of GPUs to run the training on (for CPU jobs this has to be 1)
    distributed_world_size: int = 1
    # Path to a snapshot of a trained model to keep training on
    load_snapshot_path: str = ""
    # Where to save the trained pytorch model
    save_snapshot_path: str = "/tmp/model.pt"
    # Exported caffe model will be stored here
    export_caffe2_path: str = "/tmp/model.caffe2.predictor"
    # Base directory where modules are saved
    modules_save_dir: str = ""
    # Whether to save intermediate checkpoints for modules
    save_module_checkpoints: bool = False

    # TODO these two configs are only kept only to be backward comptible with
    # RNNG, should be removed once RNNG refactoring is done
    test_out_path: str = "/tmp/test_out.txt"
    debug_path: str = "/tmp/model.debug"


class TestConfig(ConfigBase):
    # Snapshot of a trained model to test
    load_snapshot_path: str
    # Test data path
    test_path: str = "test.tsv"
    use_cuda_if_available: bool = True


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"


class OptimizerParams(ConfigBase):
    type: OptimizerType = OptimizerType.ADAM
    # Learning rate
    lr: float = 0.001
    weight_decay: float = 0.00001
    momentum: float = 0.0
