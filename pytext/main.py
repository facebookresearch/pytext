#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import pprint
import sys
import tempfile

import click
import torch
from pytext import create_predictor
from pytext.config import PyTextConfig, TestConfig
from pytext.config.serialize import Mode, config_from_json, config_to_json, parse_config
from pytext.task import load
from pytext.utils.documentation_helper import (
    ROOT_CONFIG,
    find_config_class,
    pretty_print_config_class,
)
from pytext.workflow import (
    batch_predict,
    export_saved_model_to_caffe2,
    test_model,
    train_model,
)
from torch.multiprocessing.spawn import spawn


class Attrs:
    def __repr__(self):
        return f"Attrs({', '.join(f'{k}={v}' for k, v in vars(self).items())})"


def train_model_distributed(config):
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
        run_single(0, config_to_json(PyTextConfig, config), 1, None)
    else:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".dist_sync"
        ) as sync_file:
            dist_init_method = "file://" + sync_file.name
            spawn(
                run_single,
                (
                    config_to_json(PyTextConfig, config),
                    config.distributed_world_size,
                    dist_init_method,
                ),
                config.distributed_world_size,
            )


def run_single(rank, config_json: str, world_size: int, dist_init_method: str):
    config = config_from_json(PyTextConfig, config_json)
    train_model(config, dist_init_method, rank, rank, world_size)


@click.group()
@click.option("--config-file", default="")
@click.option("--config-json", default="")
@click.pass_context
def main(context, config_file, config_json):
    """Configs can be passed by file or directly from json.
    If neither --config-file or --config-json is passed,
    attempts to read the file from stdin.

    Example:

      pytext train < demos/docnn.json
    """
    context.obj = Attrs()

    def load_config():
        if not hasattr(context.obj, "config_json"):
            if config_file:
                with open(config_file) as file:
                    config = json.load(file)
            elif config_json:
                config = json.loads(config_json)
            else:
                click.echo("No config file specified, reading from stdin")
                config = json.load(sys.stdin)
        # Cache the config object so it can be accessed multiple times
        context.obj.config_json = config
        return config

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


@main.command()
@click.pass_context
def test(context):
    """Test a trained model snapshot."""
    config_json = context.obj.load_config()
    config = parse_config(Mode.TEST, config_json)
    print("\n=== Starting testing...")
    test_model(config)


@main.command()
@click.pass_context
def train(context):
    """Train a model and save the best snapshot."""
    config_json = context.obj.load_config()
    config = parse_config(Mode.TRAIN, config_json)
    print("\n===Starting training...")
    if config.distributed_world_size == 1:
        train_model(config)
    else:
        train_model_distributed(config)
    print("\n=== Starting testing...")
    test_config = TestConfig(
        load_snapshot_path=config.save_snapshot_path,
        test_path=config.task.data_handler.test_path,
        use_cuda_if_available=config.use_cuda_if_available,
    )
    test_model(test_config)


@main.command()
@click.option("--model", help="the pytext snapshot model file to load")
@click.option("--output-path", help="where to save the exported model")
@click.pass_context
def export(context, model, output_path):
    """Convert a pytext model snapshot to a caffe2 model."""
    config = parse_config(Mode.TRAIN, context.obj.load_config())
    model = model or config.save_snapshot_path
    output_path = output_path or config.export_caffe2_path
    print(f"Exporting {model} to {output_path}")
    export_saved_model_to_caffe2(model, output_path)


@main.command()
@click.option("--exported-model", help="where to load the exported model")
@click.pass_context
def predict(context, exported_model):
    """Start a repl executing examples against a caffe2 model."""
    config = parse_config(Mode.TRAIN, context.obj.load_config())
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
