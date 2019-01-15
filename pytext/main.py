#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import pprint
import sys
import tempfile
from importlib import import_module
from pydoc import locate

import click
import torch
from pytext import create_predictor
from pytext.config import PyTextConfig
from pytext.config.component import register_tasks
from pytext.config.serialize import config_from_json, config_to_json, parse_config
from pytext.data.data_handler import CommonMetadata
from pytext.task import load
from pytext.utils.documentation_helper import (
    ROOT_CONFIG,
    eprint,
    find_config_class,
    pretty_print_config_class,
    replace_components,
)
from pytext.workflow import (
    export_saved_model_to_caffe2,
    prepare_task_metadata,
    test_model_from_snapshot_path,
    train_model,
)
from tensorboardX import SummaryWriter
from torch.multiprocessing.spawn import spawn


class Attrs:
    def __repr__(self):
        return f"Attrs({', '.join(f'{k}={v}' for k, v in vars(self).items())})"


def train_model_distributed(config, summary_writer):
    assert (
        config.use_cuda_if_available and torch.cuda.is_available()
    ) or config.distributed_world_size == 1, (
        "distributed training is only available for GPU training"
    )
    assert (
        config.distributed_world_size == 1
        or not config.task.__class__.__name__ == "DisjointMultitask.Config"
    ), "Distributed training currently not supported for DisjointMultitask"
    assert (
        config.distributed_world_size == 1
        or config.distributed_world_size <= torch.cuda.device_count()
    ), (
        f"Only {torch.cuda.device_count()} GPUs are available, "
        "{config.distributed_world_size} GPUs were requested"
    )

    print(f"\n=== Starting training, World size is {config.distributed_world_size}")
    if not config.use_cuda_if_available or not torch.cuda.is_available():
        run_single(
            0, config_to_json(PyTextConfig, config), 1, None, summary_writer, None
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
                    summary_writer,
                    metadata,
                ),
                config.distributed_world_size,
            )


def run_single(
    rank: int,
    config_json: str,
    world_size: int,
    dist_init_method: str,
    summary_writer: SummaryWriter,
    metadata: CommonMetadata,
):
    config = config_from_json(PyTextConfig, config_json)
    if rank != 0:
        summary_writer = None

    train_model(
        config, dist_init_method, rank, rank, world_size, summary_writer, metadata
    )


def gen_config_impl(task_name, options):
    # import the classes required by parameters
    requested_classes = [locate(opt) for opt in options] + [locate(task_name)]
    register_tasks(requested_classes)

    task_class_set = find_config_class(task_name)
    if not task_class_set:
        raise Exception(
            f"Unknown task class: {task_name} " "(try fully qualified class name?)"
        )
    elif len(task_class_set) > 1:
        raise Exception(f"Multiple tasks named {task_name}: {task_class_set}")

    task_class = next(iter(task_class_set))
    root = PyTextConfig(task=task_class.Config())

    # Use components listed in options instead of defaults
    for opt in options:
        replace_class_set = find_config_class(opt)
        if not replace_class_set:
            raise Exception(f"Not a component class: {opt}")
        elif len(replace_class_set) > 1:
            raise Exception(f"Multiple component named {opt}: {replace_class_set}")
        replace_class = next(iter(replace_class_set))
        found = replace_components(root, opt, set(replace_class.__bases__))
        if found:
            eprint("INFO - Applying option:", "->".join(reversed(found)), "=", opt)
            obj = root
            for k in reversed(found[1:]):
                obj = getattr(obj, k)
            if hasattr(replace_class, "Config"):
                setattr(obj, found[0], replace_class.Config())
            else:
                setattr(obj, found[0], replace_class())
        else:
            raise Exception(f"Unknown option: {opt}")
    return config_to_json(PyTextConfig, root)


@click.group()
@click.option("--config-file", default="")
@click.option("--config-json", default="")
@click.option(
    "--config-module", default="", help="python module that contains the config object"
)
@click.pass_context
def main(context, config_file, config_json, config_module):
    """Configs can be passed by file or directly from json.
    If neither --config-file or --config-json is passed,
    attempts to read the file from stdin.

    Example:

      pytext train < demos/docnn.json
    """
    context.obj = Attrs()

    def load_config():
        # Cache the config object so it can be accessed multiple times
        if not hasattr(context.obj, "config"):
            if config_module:
                context.obj.config = import_module(config_module).config
            else:
                if config_file:
                    with open(config_file) as file:
                        config = json.load(file)
                elif config_json:
                    config = json.loads(config_json)
                else:
                    click.echo("No config file specified, reading from stdin")
                    config = json.load(sys.stdin)
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
        cfg = gen_config_impl(task_name, options)
    except TypeError as ex:
        eprint(
            "ERROR - Cannot create this config",
            "because some fields don't have a default value:",
            ex,
        )
        sys.exit(-1)
    print(json.dumps(cfg, sort_keys=True, indent=2))


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
@click.pass_context
def test(context, model_snapshot, test_path, use_cuda, use_tensorboard):
    """Test a trained model snapshot.

    If model-snapshot is provided, the models and configuration will then be
    loaded from the snapshot rather than any passed config file.
    Otherwise, a config file will be loaded.
    """
    summary_writer = SummaryWriter() if use_tensorboard else None
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
        use_cuda = config.use_cuda_if_available
        print(f"Configured model snapshot {model_snapshot}")
    print("\n=== Starting testing...")
    try:
        test_model_from_snapshot_path(
            model_snapshot, use_cuda, test_path, summary_writer
        )
    finally:
        if summary_writer is not None:
            summary_writer.close()


@main.command()
@click.pass_context
def train(context):
    """Train a model and save the best snapshot."""

    config = context.obj.load_config()
    print("\n===Starting training...")
    summary_writer = SummaryWriter() if config.use_tensorboard else None
    try:
        if config.distributed_world_size == 1:
            train_model(config, summary_writer=summary_writer)
        else:
            train_model_distributed(config, summary_writer)
        print("\n=== Starting testing...")
        test_model_from_snapshot_path(
            config.save_snapshot_path,
            config.use_cuda_if_available,
            config.task.data_handler.test_path,
            summary_writer,
        )
    finally:
        if summary_writer is not None:
            summary_writer.close()


@main.command()
@click.option("--model", help="the pytext snapshot model file to load")
@click.option("--output-path", help="where to save the exported caffe2 model")
@click.option("--output-onnx-path", help="where to save the exported onnx model")
@click.pass_context
def export(context, model, output_path, output_onnx_path):
    """Convert a pytext model snapshot to a caffe2 model."""
    config = context.obj.load_config()
    model = model or config.save_snapshot_path
    output_path = output_path or config.export_caffe2_path
    output_onnx_path = output_onnx_path or config.export_onnx_path
    print(
        f"Exporting {model} to caffe2 file: {output_path} and onnx file: {output_onnx_path}"
    )
    export_saved_model_to_caffe2(model, output_path, output_onnx_path)


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
    task, train_config = load(model_file)
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


if __name__ == "__main__":
    main()
