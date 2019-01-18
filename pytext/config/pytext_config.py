#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import OrderedDict
from enum import Enum
from typing import Any, Union


class ConfigBaseMeta(type):
    def annotations_and_defaults(cls):
        annotations = OrderedDict()
        defaults = {}
        for base in reversed(cls.__bases__):
            if base is ConfigBase:
                continue
            annotations.update(getattr(base, "__annotations__", {}))
            defaults.update(getattr(base, "_field_defaults", {}))
        annotations.update(vars(cls).get("__annotations__", {}))
        defaults.update({k: getattr(cls, k) for k in annotations if hasattr(cls, k)})
        return annotations, defaults

    @property
    def __annotations__(cls):
        annotations, _ = cls.annotations_and_defaults()
        return annotations

    _field_types = __annotations__

    @property
    def _fields(cls):
        return cls.__annotations__.keys()

    @property
    def _field_defaults(cls):
        _, defaults = cls.annotations_and_defaults()
        return defaults


class ConfigBase(metaclass=ConfigBaseMeta):
    def items(self):
        return self._asdict().items()

    def _asdict(self):
        return {k: getattr(self, k) for k in type(self).__annotations__}

    def __init__(self, **kwargs):
        unspecified_fields = type(self).__annotations__.keys() - (
            (kwargs.keys() | type(self)._field_defaults.keys())
        )
        if unspecified_fields:
            raise TypeError(f"Failed to specify {unspecified_fields} for {type(self)}")
        vars(self).update(kwargs)

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            lines += f"{key}: {val}".split("\n")
        return "\n    ".join(lines)


class PlaceHolder:
    pass


class PyTextConfig(ConfigBase):
    # the actual task union types will be generated in runtime
    task: Union[PlaceHolder, Any]
    use_cuda_if_available: bool = True
    # Total Number of GPUs to run the training on (for CPU jobs this has to be 1)
    distributed_world_size: int = 1
    # Path to a snapshot of a trained model to keep training on
    load_snapshot_path: str = ""
    # Where to save the trained pytorch model
    save_snapshot_path: str = "/tmp/model.pt"
    # Exported caffe model will be stored here
    export_caffe2_path: str = "/tmp/model.caffe2.predictor"
    # Exported onnx model will be stored here
    export_onnx_path: str = "/tmp/model.onnx"
    # Base directory where modules are saved
    modules_save_dir: str = ""
    # Whether to save intermediate checkpoints for modules
    save_module_checkpoints: bool = False
    # Whether to use TensorBoard
    use_tensorboard: bool = True

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
    # Whether to use TensorBoard
    use_tensorboard: bool = True
