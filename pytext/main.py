#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import pprint
import sys
import tempfile
from importlib import import_module
from pydoc import locate
from typing import Dict, List, Optional, Union

import click
import torch
from pytext import create_predictor
from pytext.builtin_task import add_include
from pytext.common.utils import eprint
from pytext.config import ExportConfig, LATEST_VERSION, PyTextConfig
from pytext.config.component import register_tasks
from pytext.config.config_adapter import upgrade_to_latest
from pytext.config.serialize import (
    config_from_json,
    config_to_json,
    parse_config,
    pytext_config_from_json,
)
from pytext.config.utils import find_param, replace_param
from pytext.data.data_handler import CommonMetadata
from pytext.metric_reporters.channel import Channel, TensorBoardChannel
from pytext.PreprocessingMap.ttypes import ModelType
from pytext.task import load
from pytext.utils.documentation import (
    find_config_class,
    get_subclasses,
    pretty_print_config_class,
    replace_components,
    ROOT_CONFIG,
)
from pytext.utils.file_io import PathManager
from pytext.workflow import (
    export_saved_model_to_caffe2,
    export_saved_model_to_torchscript,
    get_logits as workflow_get_logits,
    prepare_task_metadata,
    save_pytext_snapshot as workflow_save_pytext_snapshot,
    test_model_from_snapshot_path,
    train_model,
)
from torch.multiprocessing.spawn import spawn


class Attrs:
    def __repr__(self):
        return f"Attrs({', '.join(f'{k}={v}' for k, v in vars(self).items())})"


def _validate_export_json_config(export_json_config):
    """Validate if the input export_json_config (PyTextConfig in JSON object) only has
    export section config and a version number.
    """
    assert (export_json_config.keys() <= {"export", "version", "read_chunk_size"}) or (
        export_json_config.keys() <= {"export_list", "version", "read_chunk_size"}
    ), (
        "The export-json config should only contain fields (export or export_list),  version and read_chunk_size. Got "
        f"{export_json_config.keys()}"
    )

    if "export" in export_json_config.keys():
        for key in export_json_config["export"]:
            assert (
                key in ExportConfig.__annotations__.keys()
            ), f"Field {key} in the export json is not found in the ExportConfig class."
    else:  # export_list instead of export
        assert "export_list" in export_json_config.keys()
        found_model_type = None
        export_cfgs = export_json_config["export_list"]
        for export_config in export_cfgs:
            for key in export_config:
                assert (
                    key in ExportConfig.__annotations__.keys()
                ), f"Field {key} in the export json is not found in the ExportConfig class."
            this_model_type = (
                ModelType.PYTORCH
                if "export_pytorch_path" in export_config
                else ModelType.CAFFE2
            )
            assert (found_model_type is None) or (found_model_type == this_model_type)
            if found_model_type is None:
                found_model_type = this_model_type


def _load_and_validate_export_json_config(export_json):
    with PathManager.open(export_json) as fp:
        export_json_config = json.load(fp)
        if "config" in export_json_config:
            export_json_config = export_json_config["config"]
        export_json_config = upgrade_to_latest(export_json_config)
        _validate_export_json_config(export_json_config)
        return export_json_config


def train_model_distributed(config, metric_channels: Optional[List[Channel]]):
    assert (
        config.use_cuda_if_available and torch.cuda.is_available()
    ) or config.distributed_world_size == 1, (
        "distributed training is only available for GPU training"
    )
    assert (
        config.distributed_world_size == 1
        or config.distributed_world_size <= torch.cuda.device_count()
    ), (
        f"Only {torch.cuda.device_count()} GPUs are available, "
        f"{config.distributed_world_size} GPUs were requested"
    )

    print(f"\n=== Starting training, World size is {config.distributed_world_size}")
    if not config.use_cuda_if_available or not torch.cuda.is_available():
        run_single(
            rank=0,
            config_json=config_to_json(PyTextConfig, config),
            world_size=1,
            dist_init_method=None,
            metadata=None,
            metric_channels=metric_channels,
        )
    else:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".dist_sync"
        ) as sync_file:
            dist_init_method = "file://" + sync_file.name
            metadata = prepare_task_metadata(config)
            spawn(
                run_single,
                (
                    config_to_json(PyTextConfig, config),
                    config.distributed_world_size,
                    dist_init_method,
                    metadata,
                    [],
                ),
                config.distributed_world_size,
            )


def run_single(
    rank: int,
    config_json: str,
    world_size: int,
    dist_init_method: Optional[str],
    metadata: Optional[Union[Dict[str, CommonMetadata], CommonMetadata]],
    metric_channels: Optional[List[Channel]],
):
    config = pytext_config_from_json(config_json)
    if rank != 0:
        metric_channels = []

    train_model(
        config=config,
        dist_init_url=dist_init_method,
        device_id=rank,
        rank=rank,
        world_size=world_size,
        metric_channels=metric_channels,
        metadata=metadata,
    )


def gen_config_impl(task_name, *args, **kwargs):
    # import the classes required by parameters
    requested_classes = [locate(opt) for opt in args] + [locate(task_name)]
    register_tasks(requested_classes)

    task_class_set = find_config_class(task_name)
    if not task_class_set:
        raise Exception(
            f"Unknown task class: {task_name} " "(try fully qualified class name?)"
        )
    elif len(task_class_set) > 1:
        raise Exception(f"Multiple tasks named {task_name}: {task_class_set}")

    task_class = next(iter(task_class_set))
    task_config = getattr(task_class, "example_config", task_class.Config)
    root = PyTextConfig(task=task_config(), version=LATEST_VERSION)
    eprint("INFO - Applying task option:", task_class.__name__)

    # Use components in args instead of defaults
    for opt in args:
        if "=" in opt:
            param_path, value = opt.split("=", 1)
            kwargs[param_path] = value
            continue
        replace_class_set = find_config_class(opt)
        if not replace_class_set:
            raise Exception(f"Not a component class: {opt}")
        elif len(replace_class_set) > 1:
            raise Exception(f"Multiple component named {opt}: {replace_class_set}")
        replace_class = next(iter(replace_class_set))
        found = replace_components(root, opt, get_subclasses(replace_class))
        if found:
            eprint("INFO - Applying class option:", ".".join(reversed(found)), "=", opt)
            obj = root
            for k in reversed(found[1:]):
                obj = getattr(obj, k)
            if hasattr(replace_class, "Config"):
                setattr(obj, found[0], replace_class.Config())
            else:
                setattr(obj, found[0], replace_class())
        else:
            raise Exception(f"Unknown class option: {opt}")

    # Use parameters in kwargs instead of defaults
    for param_path, value in kwargs.items():
        found = find_param(root, "." + param_path)
        if len(found) == 1:
            eprint("INFO - Applying parameter option to", found[0], "=", value)
            replace_param(root, found[0].split("."), value)
        elif not found:
            raise Exception(f"Unknown parameter option: {param_path}")
        else:
            raise Exception(
                f"Multiple possibilities for {param_path}: {', '.join(found)}"
            )

    return root


@click.group()
@click.option(
    "--include", multiple=True, help="directory containing custom python classes"
)
@click.option("--config-file", default="")
@click.option("--config-json", default="")
@click.option(
    "--config-module", default="", help="python module that contains the config object"
)
@click.pass_context
def main(context, config_file, config_json, config_module, include):
    """Configs can be passed by file or directly from json.
    If neither --config-file or --config-json is passed,
    attempts to read the file from stdin.

    Example:

      pytext train < demo/configs/docnn.json
    """
    for path in include or []:
        # remove possible trailing / from autocomplete in --include
        add_include(path.rstrip("/"))

    context.obj = Attrs()
    context.obj.include = include

    def load_config():
        # Cache the config object so it can be accessed multiple times
        if not hasattr(context.obj, "config"):
            if config_module:
                context.obj.config = import_module(config_module).config
            else:
                if config_file:
                    with PathManager.open(config_file) as fp:
                        config = json.load(fp)
                elif config_json:
                    config = json.loads(config_json)
                else:
                    eprint("No config file specified, reading from stdin")
                    config = json.load(sys.stdin)
                # before parsing the config, include the custom components
                for path in config.get("include_dirs", None) or []:
                    add_include(path.rstrip("/"))
                context.obj.config = parse_config(config)
        return context.obj.config

    context.obj.load_config = load_config


@main.command(help="Print help information on a config parameter")
@click.argument("class_name", default=ROOT_CONFIG)
@click.pass_context
def help_config(context, class_name):
    """
    Find all the classes matching `class_name`, and
    pretty-print each matching class field members (non-recursively).
    """
    found_classes = find_config_class(class_name)
    if found_classes:
        for obj in found_classes:
            pretty_print_config_class(obj)
            print()
    else:
        raise Exception(f"Unknown component name: {class_name}")


@main.command(help="Generate a config JSON file with default values.")
@click.argument("task_name")
@click.argument("options", nargs=-1)
@click.pass_context
def gen_default_config(context, task_name, options):
    """
    Generate a config for `task_name` with default values.
    Optionally, override the defaults by passing your desired
    components as `options`.
    """
    try:
        cfg = gen_config_impl(task_name, *options)
    except TypeError as ex:
        eprint(
            "ERROR - Cannot create this config",
            "because some fields don't have a default value:",
            ex,
        )
        sys.exit(-1)

    # add the --include to the config generated
    if context.obj.include:
        if cfg.include_dirs is None:
            cfg.include_dirs = []
        for path in context.obj.include:
            cfg.include_dirs.append(path.rstrip("/"))

    cfg_json = config_to_json(PyTextConfig, cfg)
    print(json.dumps(cfg_json, sort_keys=True, indent=2))


@main.command(help="Update a config JSON file to lastest version.")
@click.pass_context
def update_config(context):
    """
    Load a config file, update to latest version and prints the result.
    """
    config = context.obj.load_config()
    config_json = config_to_json(PyTextConfig, config)
    print(json.dumps(config_json, sort_keys=True, indent=2))


@main.command()
@click.option(
    "--model-snapshot",
    default="",
    help="load model snapshot and test configuration from this file",
)
@click.option("--test-path", default="", help="path to test data")
@click.option(
    "--use-cuda/--no-cuda",
    default=None,
    help="Run supported parts of the model on GPU if available.",
)
@click.option(
    "--use-tensorboard/--no-tensorboard",
    default=True,
    help="Whether to visualize test metrics using TensorBoard.",
)
@click.option(
    "--field_names",
    default=None,
    help="""Field names for the test-path. If this is not set, the first line of
         each file will be assumed to be a header containing the field names.""",
)
@click.pass_context
def test(context, model_snapshot, test_path, use_cuda, use_tensorboard, field_names):
    """Test a trained model snapshot.

    If model-snapshot is provided, the models and configuration will then be
    loaded from the snapshot rather than any passed config file.
    Otherwise, a config file will be loaded.
    """
    model_snapshot, use_cuda, use_tensorboard = _get_model_snapshot(
        context, model_snapshot, use_cuda, use_tensorboard
    )
    print("\n=== Starting testing...")
    metric_channels = []
    if use_tensorboard:
        metric_channels.append(TensorBoardChannel())
    try:
        test_model_from_snapshot_path(
            model_snapshot,
            use_cuda,
            test_path,
            metric_channels,
            field_names=field_names,
        )
    finally:
        for mc in metric_channels:
            mc.close()


def _get_model_snapshot(context, model_snapshot, use_cuda, use_tensorboard):
    if model_snapshot:
        print(f"Loading model snapshot and config from {model_snapshot}")
        if use_cuda is None:
            raise Exception(
                "if --model-snapshot is set --use-cuda/--no-cuda must be set"
            )
    else:
        print(f"No model snapshot provided, loading from config")
        config = context.obj.load_config()
        model_snapshot = config.save_snapshot_path
        use_cuda = config.use_cuda_if_available and getattr(
            config, "use_cuda_for_testing", True
        )
        use_tensorboard = config.use_tensorboard
        print(f"Configured model snapshot {model_snapshot}")
    return model_snapshot, use_cuda, use_tensorboard


@main.command()
@click.pass_context
def train(context):
    """Train a model and save the best snapshot."""

    config = context.obj.load_config()
    print("\n===Starting training...")
    metric_channels = []
    if config.use_tensorboard:
        metric_channels.append(TensorBoardChannel())
    try:
        if config.distributed_world_size == 1:
            train_model(config, metric_channels=metric_channels)
        else:
            train_model_distributed(config, metric_channels)
        print("\n=== Starting testing...")
        test_model_from_snapshot_path(
            config.save_snapshot_path,
            config.use_cuda_if_available,
            test_path=None,
            metric_channels=metric_channels,
        )
    finally:
        for mc in metric_channels:
            mc.close()


@main.command()
@click.option("--export-json", help="the path to the export options in JSON format.")
@click.option("--model", help="the pytext snapshot model file to load")
@click.option("--output-path", help="where to save the exported caffe2 model")
@click.option("--output-onnx-path", help="where to save the exported onnx model")
@click.pass_context
def export(context, export_json, model, output_path, output_onnx_path):
    """Convert a pytext model snapshot to a caffe2 model."""
    # only populate from export_json if no export option is configured from the command line.
    if export_json:
        if not output_path and not output_onnx_path:
            export_json_config = _load_and_validate_export_json_config(export_json)
            export_section_config = export_json_config["export"]
            if "export_caffe2_path" in export_section_config:
                output_path = export_section_config["export_caffe2_path"]
            if "export_onnx_path" in export_section_config:
                output_onnx_path = export_section_config["export_onnx_path"]
        else:
            print(
                "the export-json config is ignored because export options are found the command line"
            )
    config = context.obj.load_config()
    model = model or config.save_snapshot_path
    if config.export:
        output_path = output_path or config.export_caffe2_path
        output_onnx_path = output_onnx_path or config.export_onnx_path
        print(
            f"Exporting {model} to caffe2 file: {output_path} and onnx file: {output_onnx_path}"
        )
        export_saved_model_to_caffe2(model, output_path, output_onnx_path)
    else:
        for idx in range(0, len(config.export_list)):
            output_path = output_path or config.get_export_caffe2_path(idx)
            output_onnx_path = output_onnx_path or config.get_export_onnx_path(idx)
            print(
                f"Exporting {model} to caffe2 file: {output_path} and onnx file: {output_onnx_path}"
            )
            export_saved_model_to_caffe2(model, output_path, output_onnx_path)


@main.command()
@click.option("--export-json", help="the path to the export options in JSON format.")
@click.option("--model", help="the pytext snapshot model file to load")
@click.option("--output-path", help="where to save the exported torchscript model")
@click.option("--quantize", help="whether to quantize the model")
@click.option("--target", help="specify the name of a single model to export")
@click.pass_context
def torchscript_export(context, export_json, model, output_path, quantize, target):
    """Convert a pytext model snapshot to a torchscript model."""
    export_cfg = ExportConfig()
    # only populate from export_json if no export option is configured from the command line.
    if export_json:
        export_json_config = _load_and_validate_export_json_config(export_json)

        read_chunk_size = export_json_config.pop("read_chunk_size", None)
        if read_chunk_size is not None:
            print("Warning: Ignoring read_chunk_size.")

        if export_json_config.get("read_chunk_size", None) is not None:
            print("Error: Do not know what to do with read_chunk_size.  Ignoring.")

        if "export" in export_json_config.keys():
            export_cfgs = [export_json_config["export"]]
        else:
            export_cfgs = export_json_config["export_list"]

        if target:
            print(
                "A single export was specified in the command line. Filtering out all other export options"
            )
            export_cfgs = [cfg for cfg in export_cfgs if cfg["target"] == target]
            if export_cfgs == []:
                print(
                    "No ExportConfig matches the target name specified in the command line."
                )

        for partial_export_cfg in export_cfgs:
            if not quantize and not output_path:
                export_cfg = config_from_json(ExportConfig, partial_export_cfg)
            else:
                print(
                    "the export-json config is ignored because export options are found the command line"
                )
                export_cfg = config_from_json(
                    ExportConfig,
                    partial_export_cfg,
                    ("export_caffe2_path", "export_onnx_path"),
                )
                export_cfg.torchscript_quantize = quantize
            # if config has export_torchscript_path, use export_torchscript_path from config, otherwise keep the default from CLI
            if export_cfg.export_torchscript_path is not None:
                output_path = export_cfg.export_torchscript_path
            if not model or not output_path:
                config = context.obj.load_config()
                model = model or config.save_snapshot_path
                output_path = output_path or f"{config.save_snapshot_path}.torchscript"

            print(f"Exporting {model} to torchscript file: {output_path}")
            print(export_cfg)
            export_saved_model_to_torchscript(model, output_path, export_cfg)


@main.command()
@click.option("--exported-model", help="where to load the exported model")
@click.pass_context
def predict(context, exported_model):
    """Start a repl executing examples against a caffe2 model."""
    config = context.obj.load_config()
    print(f"Loading model from {exported_model or config.export_caffe2_path}")
    predictor = create_predictor(config, exported_model)

    print(f"Model loaded, reading example JSON from stdin")
    for line in sys.stdin.readlines():
        input = json.loads(line)
        predictions = predictor(input)
        pprint.pprint(predictions)


@main.command()
@click.option("--model-file", help="where to load the pytorch model")
@click.pass_context
def predict_py(context, model_file):
    """
    Start a repl executing examples against a PyTorch model.
    Example is in json format with names being the same with column_to_read
    in model training config
    """
    task, train_config, _training_state = load(model_file)
    while True:
        try:
            line = input(
                "please input a json example, the names should be the same with "
                + "column_to_read in model training config: \n"
            )
            if line:
                pprint.pprint(task.predict([json.loads(line)])[0])
        except EOFError:
            break


@main.command()
@click.option(
    "--model-snapshot",
    default="",
    help="load model snapshot and test configuration from this file",
)
@click.option("--test-path", default="", help="path to test data")
@click.option("--output-path", default="", help="path to save logits")
@click.option(
    "--use-cuda/--no-cuda",
    default=None,
    help="Run supported parts of the model on GPU if available.",
)
@click.option(
    "--field_names",
    default=None,
    help="""Field names for the test-path. If this is not set, the first line of
         each file will be assumed to be a header containing the field names.""",
)
@click.option(
    "--dump-raw-input/--no-dump-raw-input",
    default=False,
    help="Store the input data as a column in the output file.",
)
@click.option(
    "--batch-size",
    default=16,
    show_default=True,
    help="The batch size. Bigger batch sizes lead to better GPU utlization",
)
@click.option(
    "--ndigits-precision",
    default=0,
    show_default=True,
    help="""The digists precision of serialized floats.
    The default 0 means don't round float and results a larger output file.""",
)
@click.option(
    "--output-columns",
    type=str,
    default=None,
    help="""If the model returns mutliple outputs, only the output-columns will be kept.
    Takes a comma separated list of integers. By default all outputs are written.""",
)
@click.option(
    "--use-gzip/--no-gzip",
    default=False,
    help="Using gzip significantly reduces the output size by 3-4x",
)
@click.option("--device-id", default=0, show_default=True, help="""CUDA device id.""")
@click.pass_context
def get_logits(
    context,
    model_snapshot,
    test_path,
    use_cuda,
    output_path,
    field_names,
    dump_raw_input,
    batch_size,
    ndigits_precision,
    output_columns,
    use_gzip,
    device_id,
):
    """print logits from  a trained model snapshot to output_path"""

    model_snapshot, use_cuda, _ = _get_model_snapshot(
        context, model_snapshot, use_cuda, False
    )
    if output_columns:
        output_columns = [int(x) for x in output_columns.split(",")]

    print("\n=== Starting get_logits...")
    workflow_get_logits(
        model_snapshot,
        use_cuda,
        output_path,
        test_path,
        field_names,
        dump_raw_input,
        batch_size,
        ndigits_precision,
        output_columns,
        use_gzip,
        device_id,
    )


@main.command()
@click.pass_context
def save_pytext_snapshot(context):
    """Load a PyText task and save snapshot for later use.
    This is helpful when you want to plug in a pretrained encoder in a PyText
    task and either test or generate logits using the task.
    """
    config = context.obj.load_config()
    workflow_save_pytext_snapshot(config)


if __name__ == "__main__":
    main()
