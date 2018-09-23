#!/usr/bin/env python3
from enum import Enum
from typing import Any, List, NamedTupleMeta, Optional, Tuple, Union  # noqa

from pytext.common.python_utils import InheritableNamedTupleMeta


class ConfigBase(metaclass=InheritableNamedTupleMeta):
    _root = True


class JobSpecPlaceHolder:
    pass


class PyTextConfig(ConfigBase):
    # the actual task union types will be generated in runtime
    jobspec: Union[Any, JobSpecPlaceHolder]
    use_cuda_if_available: bool = True

    train_path: str = "train.tsv"
    eval_path: str = "eval.tsv"
    test_path: str = "test.tsv"
    # Training batch_size
    train_batch_size: int = 128
    # Eval batch_size
    eval_batch_size: int = 128
    # Test batch size
    test_batch_size: int = 128
    # A path to a snapshot of a trained model to test
    load_snapshot_path: str = ""
    # A file to store the output of the model when running on the test data
    test_out_path: str = "/tmp/test_out.txt"
    # Where to save the trained pytorch model
    save_snapshot_path: str = "/tmp/model.pt"
    # A file to store model debug information
    debug_path: str = "/tmp/model.debug"
    # Exported caffe model will be stored here
    export_caffe2_path: str = "/tmp/model.caffe2.predictor"
    # if test only TODO, better to use explicit "train" "test" cmd
    test_given_snapshot: bool = False


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"


class OptimizerParams(ConfigBase):
    type: OptimizerType = OptimizerType.ADAM
    # Learning rate
    lr: float = 0.001
    weight_decay: float = 0.00001
    momentum: float = 0.0


class SchedulerType(Enum):
    NONE = "none"
    STEP_LR = "step_lr"
    EXPONENTIAL_LR = "exponential_lr"
    COSINE_ANNEALING_LR = "cosine_annealing_lr"


class SchedulerParams(ConfigBase):
    type: SchedulerType = SchedulerType.NONE
    step_size: int = 1000
    gamma: float = 0.1
    T_max: int = 1000
    eta_min: float = 0
