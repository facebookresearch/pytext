#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import OrderedDict
from typing import Any, List, Optional, Union


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
        """Configs can be constructed by specifying values by keyword.
        If a keyword is supplied that isn't in the config, or if a config requires
        a value that isn't specified and doesn't have a default, a TypeError will be
        raised."""
        specified = kwargs.keys() | type(self)._field_defaults.keys()
        required = type(self).__annotations__.keys()
        # Unspecified fields have no default and weren't provided by the caller
        unspecified_fields = required - specified
        if unspecified_fields:
            raise TypeError(f"Failed to specify {unspecified_fields} for {type(self)}")

        # Overspecified fields are fields that were provided but that the config
        # doesn't know what to do with, ie. was never specified anywhere.
        overspecified_fields = specified - required
        if overspecified_fields:
            raise TypeError(
                f"Specified non-existent fields {overspecified_fields} for {type(self)}"
            )

        vars(self).update(kwargs)

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in sorted(self._asdict().items()):
            lines += f"{key}: {val}".split("\n")
        return "\n    ".join(lines)


class PlaceHolder:
    pass


class PyTextConfig(ConfigBase):
    # the actual task union types will be generated in runtime
    task: Union[PlaceHolder, Any]
    use_cuda_if_available: bool = True
    # Enable mixed precision training. WARNING: under develoment
    use_fp16: bool = False
    # Total Number of GPUs to run the training on (for CPU jobs this has to be 1)
    distributed_world_size: int = 1
    # Path to a snapshot of a trained model to keep training on
    load_snapshot_path: str = ""
    # Where to save the trained pytorch model
    save_snapshot_path: str = "/tmp/model.pt"
    # Exported caffe model will be stored here
    export_caffe2_path: Optional[str] = "/tmp/model.caffe2.predictor"
    # Exported onnx model will be stored here
    export_onnx_path: str = "/tmp/model.onnx"
    # Exported torchscript model will be stored here
    export_torchscript_path: Optional[str] = None
    # Base directory where modules are saved
    modules_save_dir: str = ""
    # Whether to save intermediate checkpoints for modules if they are best yet
    save_module_checkpoints: bool = False
    # Whether to save ALL intermediate checkpoints for modules
    save_all_checkpoints: bool = False
    # Whether to use TensorBoard
    use_tensorboard: bool = True
    #: Seed value to seed torch, python, and numpy random generators.
    random_seed: Optional[int] = None
    #: Whether to allow CuDNN to behave deterministically.
    use_deterministic_cudnn: bool = False
    # Run eval set after model has been trained - for hyperparameter search
    report_eval_results: bool = False
    # config version
    version: int

    # TODO these two configs are only kept only to be backward comptible with
    # RNNG, should be removed once RNNG refactoring is done
    test_out_path: str = "/tmp/test_out.txt"
    debug_path: str = "/tmp/model.debug"


class TestConfig(ConfigBase):
    # Snapshot of a trained model to test
    load_snapshot_path: str
    # Test data path
    test_path: Optional[str] = "test.tsv"
    #: Field names for the TSV. If this is not set, the first line of each file
    #: will be assumed to be a header containing the field names.
    field_names: Optional[List[str]] = None
    use_cuda_if_available: bool = True
    # Whether to use TensorBoard
    use_tensorboard: bool = True
    # Output path where metric reporter writes to.
    test_out_path: str = ""


LATEST_VERSION = 12
