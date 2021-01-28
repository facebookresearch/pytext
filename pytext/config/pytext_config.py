#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
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

    def _replace(self, **kwargs):
        args = self._asdict()
        args.update(kwargs)
        return type(self)(**args)

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

    def __eq__(self, other):
        """Mainly a convenience utility for unit testing."""
        return type(self) == type(other) and self._asdict() == other._asdict()


class PlaceHolder:
    pass


class ExportConfig(ConfigBase):
    # Exported caffe model will be stored here
    export_caffe2_path: Optional[str] = None
    # Exported onnx model will be stored here
    export_onnx_path: str = "/tmp/model.onnx"
    # Exported torchscript model will be stored here
    export_torchscript_path: Optional[str] = None
    # Exported jit lite model will be stored here
    export_lite_path: Optional[str] = None
    # Export quantized torchscript model
    torchscript_quantize: Optional[bool] = False
    # Accelerator options.
    # Options:
    # "half" - demote model to half precision
    # "nnpi" - freeze model for use with Glow on NNPI accelerator
    accelerate: List[str] = []
    # Inference Interface.
    # Specifies which of the 3 optional list parameters a model takes,
    # when the model implements the inference_ionterface() method.:
    # Possible values: texts, multi_texts, tokens (and/or others as
    # supported by inference_interface method).
    inference_interface: Optional[str] = None
    # Padding boundaries for padded tensor sequence length dimension.
    # Specified as a list of boundaries to be rounded up to.
    # Each batch seq length dimension will be rounded to the smallest number
    # larger than the actual longest sequence in a batch.
    # The list of padding boundaries must be sorted in asecending order.
    # The first list element must be 0.  (Will serve as future padding control "version number")
    seq_padding_control: Optional[List[int]] = None
    # Padding boundaries for padded tensor batch length dimension.
    # Specified as a list of boundaries to be rounded up to.
    # Each batch length dimension will be rounded to the smallest number
    # larger than the actual longest sequence in a batch.
    # The list of padding boundaries must be sorted in asecending order.
    # The first list element must be 0.  (Will serve as future padding control "version number")
    batch_padding_control: Optional[List[int]] = None


class InvalidMethodInvocation(Exception):
    message: str

    def __init__(self, message):
        self.message = message


class PyTextConfig(ConfigBase):
    # the actual task union types will be generated in runtime
    task: Union[PlaceHolder, Any]
    use_cuda_if_available: bool = True
    # Enable mixed precision training. WARNING: under develoment
    use_fp16: bool = False
    # Total Number of GPUs to run the training on (for CPU jobs this has to be 1)
    distributed_world_size: int = 1
    # Total number of GPU streams for gradient sync in distributed training
    gpu_streams_for_distributed_training: int = 1
    # load either model or checkpoint(model + config + training_state etc)
    # load model file for inference only, load checkpont file to continue training
    load_snapshot_path: str = ""
    # Where to save the trained pytorch model and checkpoints
    save_snapshot_path: str = "/tmp/model.pt"
    # True: use the config saved in snapshot. False: use config from current task
    use_config_from_snapshot: bool = True
    # if there are existing snapshots in parent directory of save_snapshot_path
    # resume training from the latest snapshot automatically
    auto_resume_from_snapshot: bool = False
    # Configuration for model export. See ExportConfig for details
    export: ExportConfig = ExportConfig()
    # Configuration for a list of model exports. If the list is non-empty, export will be ignored.
    export_list: List[ExportConfig] = []
    # Base directory where modules are saved
    modules_save_dir: str = ""
    # Whether to save intermediate checkpoints for modules if they are best yet
    save_module_checkpoints: bool = False
    # Whether to save ALL intermediate checkpoints for modules
    save_all_checkpoints: bool = False
    # Whether to use TensorBoard
    use_tensorboard: bool = True
    #: Seed value to seed torch, python, and numpy random generators.
    random_seed: Optional[int] = 0
    #: Whether to allow CuDNN to behave deterministically.
    use_deterministic_cudnn: bool = False
    # Run eval set after model has been trained - for hyperparameter search
    report_eval_results: bool = False
    # Run test set after model has been trained
    report_test_results: bool = True
    # include components from custom directories
    include_dirs: Optional[List[str]] = None
    # config version
    version: int
    # Use CUDA for testing. Set to false for models where testing on CPU is
    # preferred. This option allows one to train on GPU and test on CPU by
    # setting use_cuda_if_available=True and use_cuda_for_testing=False. Note
    # that if use_cuda_if_available=False or CUDA is not available, this
    # parameter has no effect.
    use_cuda_for_testing: bool = True

    # When reading large files(in manifold), PathManager will read by chunks
    # The memory usage can be estimated by: read_chunk_size * num_process
    # If you got Out-of-Memory issue due to using many GPUs(1 process/GPU),
    # you can decrease read_chunk_size to reduce memory usage
    read_chunk_size: Optional[int] = 1000 ** 3  # 1GB

    # TODO these two configs are only kept only to be backward comptible with
    # RNNG, should be removed once RNNG refactoring is done
    test_out_path: str = "/tmp/test_out.txt"
    debug_path: str = "/tmp/model.debug"

    def __init__(self, **kwargs):
        version = kwargs["version"]
        if version < 22:
            assert "export" not in kwargs, (
                'Config versions before 22 should not contain an "export" section. Got '
                f"version={version}."
            )
            kwargs["export"] = ExportConfig(
                **{
                    k: kwargs.pop(k)
                    for k in ExportConfig.__annotations__.keys()
                    if k in kwargs.keys()
                }
            )
            kwargs["export_list"] = [kwargs["export"]]
            kwargs["version"] = 22
        super().__init__(**kwargs)
        if len(self.export_list) == 0:  # Happens if version >= 22:
            self.export_list = [self.export]

    def export_check(self, method_name):
        if len(self.export_list) != 1:
            if len(self.export_list) == 0:
                # Is there a proper finalizer that can be called instead??
                # Need help from a python Guru
                self.export_list = [self.export]
            else:
                raise InvalidMethodInvocation(
                    "export list length is not 1  use the set/get_%s version of method with key"
                    % (method_name,)
                )

    @property
    def export_caffe2_path(self):
        self.export_check("export_caffe2_path")
        return self.export_list[0].export_caffe2_path

    @export_caffe2_path.setter
    def export_caffe2_path(self, p):
        self.export_check("export_caffe2_path")
        self.export_list[0].export_caffe2_path = p

    def get_export_caffe2_path(self, index):
        return self.export_list[index].export_caffe2_path

    def set_export_caffe2_path(self, p, index):
        self.export_list[index].export_caffe2_path = p

    @property
    def export_onnx_path(self):
        self.export_check("export_onnx_path")
        return self.export_list[0].export_onnx_path

    @export_onnx_path.setter
    def export_onnx_path(self, p):
        self.export_check("export_onnx_path")
        self.export_list[0].export_onnx_path = p

    def get_export_onnx_path(self, index):
        return self.export_list[index].export_onnx_path

    def set_export_onnx_path(self, p, index):
        self.export_list[index].export_onnx_path = p

    @property
    def export_torchscript_path(self):
        self.export_check("export_torchscript_path")
        return self.export_list[0].export_torchscript_path

    @export_torchscript_path.setter
    def export_torchscript_path(self, p):
        self.export_check("export_torchscript_path")
        self.export_list[0].export_torchscript_path = p

    def get_export_torchscript_path(self, index):
        return self.export_list[index].export_torchscript_path

    def set_export_torchscript_path(self, p, index):
        self.export_list[index].export_torchscript_path = p

    @property
    def torchscript_quantize(self):
        self.export_check("torchscript_quantize")
        return self.export_list[0].torchscript_quantize

    @torchscript_quantize.setter
    def torchscript_quantize(self, quantize):
        self.export_check("torchscript_quantize")
        self.export_list[0].torchscript_quantize = quantize

    def get_export_torchscript_quantize(self, index):
        return self.export_list[index].torchscript_quantize

    def set_export_torchscript_path(self, quantize, index):
        self.export_list[index].torchscript_quantize = quantize

    @property
    def accelerate(self):
        self.export_check("accelerate")
        return self.export_list[0].accelerate

    @accelerate.setter
    def accelerate(self, acc):
        self.export_check("accelerate")
        self.export_list[0].accelerate = acc

    def get_export_accelerate(self, index):
        return self.export_list[index].accelerate

    def set_export_accelerate(self, acc, index):
        self.export_list[index].accelerate = acc

    @property
    def inference_interface(self):
        self.export_check("inference_interface")
        return self.export_list[0].inference_interface

    @inference_interface.setter
    def inference_interface(self, inf_inter):
        self.export_check("inference_interface")
        self.export_list[0].inference_interface = inf_inter

    def get_export_inference_interface(self, index):
        return self.export_list[index].inference_interface

    def set_export_inference_interface(self, inference_interface, index):
        self.export_list[index].inference_interface = inference_interface

    @property
    def seq_padding_control(self):
        self.export_check("seq_padding_control")
        return self.export_list[0].seq_padding_control

    @seq_padding_control.setter
    def seq_padding_control(self, spc):
        self.export_check("seq_padding_control")
        self.export_list[0].seq_padding_control = spc

    def get_export_seq_padding_control(self, index):
        return self.export_list[index].seq_padding_control

    def set_export_inference_interface(self, seq_padding_control, index):
        self.export_list[index].seq_padding_control = seq_padding_control

    @property
    def batch_padding_control(self):
        self.export_check("batch_padding_control")
        return self.export_list[0].batch_padding_control

    @batch_padding_control.setter
    def batch_padding_control(self, bpc):
        self.export_check("batch_padding_control")
        self.export_list[0].batch_padding_control = bpc

    def get_export_batch_padding_control(self, index):
        return self.export_list[index].batchpadding_control

    def set_export_batch_padding_control(self, batch_padding_control, index):
        self.export_list[index].batch_padding_control = batch_padding_control


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
    # Enable mixed precision training. WARNING: under develoment
    use_fp16: bool = False


class LogitsConfig(TestConfig):
    # List of test data paths
    gpus: int = 1
    # Whether to dump the raw input to output file.
    dump_raw_input: bool = False
    # The batch size. Bigger batch sizes lead to better GPU utlization
    batch_size: int = 16
    # The digists precision of serialized floats.
    # The default 0 means don't round float and results a larger output file
    ndigits_precision: int = 0
    # If the model returns mutliple outputs, only the output-columns will be kept.
    # By default all outputs are written
    output_columns: Optional[List[int]] = None
    # Usign gzip significantly reduces the output size by 3-4x
    use_gzip: bool = False
    # Use fp16 for inference
    fp16: bool = False


# update sitevar PYTEXT_CONFIG_LATEST_VERSION when new PytextConfig pushed in pytext config
LATEST_VERSION = 24
