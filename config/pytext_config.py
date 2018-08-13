#!/usr/bin/env python3
from enum import Enum
from collections import OrderedDict
from typing import Any, List, NamedTupleMeta, Optional, Union, Tuple # noqa


class ConfigMeta(NamedTupleMeta):
    def __new__(cls, typename, bases, ns):
        annotations = OrderedDict()
        annotations.update(ns.get("__annotations__", {}))
        for base in bases:
            if not issubclass(base, Tuple):
                continue
            base_fields = getattr(base, "__annotations__", {})

            for field_name in base_fields:
                # pass along default values
                if field_name in base._field_defaults and field_name not in ns:
                    ns[field_name] = base._field_defaults[field_name]
                if field_name not in annotations:
                    annotations[field_name] = base_fields[field_name]
                    if field_name not in ns:
                        annotations.move_to_end(field_name, last=False)
        if len(annotations) == 0:
            # fbl flow types don't support empty namedTuple,
            # add placeholder to workaround
            annotations["config_name_"] = str
            ns["config_name_"] = typename
        ns["__annotations__"] = annotations
        return super().__new__(cls, typename, bases, ns)


class ConfigBase(metaclass=ConfigMeta):
    _root = True


class JobSpecPlaceHolder:
    pass


class PyTextConfig(ConfigBase):
    # the actual task union types will be generated in runtime
    jobspec: Union[Any, JobSpecPlaceHolder]
    use_cuda_if_available: bool = True

    train_file_path: str = "train.tsv"
    eval_file_path: str = "eval.tsv"
    test_file_path: str = "test.tsv"
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
    export_caffe2_path: str = "/tmp/model.fbl.predictor"
    # if test only TODO, better to use explicit "train" "test" cmd
    test_given_snapshot: bool = False


class OptimizerType(Enum):
    ADAM = 'adam'
    SGD = 'sgd'


class OptimizerParams(ConfigBase):
    type: OptimizerType = OptimizerType.ADAM
    # Learning rate
    lr: float = 0.001
    weight_decay: float = 0.00001
    momentum: float = 0.0
