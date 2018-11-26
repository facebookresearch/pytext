#!/usr/bin/env python3

import enum
import json
import sys
import tempfile

import click
import torch
from pytext.config import PyTextConfig, TestConfig
from pytext.config.serialize import config_from_json, config_to_json
from pytext.workflow import test_model, train_model
from torch.multiprocessing.spawn import spawn


class Mode(enum.Enum):
    TRAIN = "train"
    TEST = "test"


def parse_config(mode, config_json):
    """
    Parse PyTextConfig object from parameter string or parameter file
    """
    config_cls = {Mode.TRAIN: PyTextConfig, Mode.TEST: TestConfig}[mode]
    # TODO T32608471 should assume the entire json is PyTextConfig later, right
    # now we're matching the file format for pytext trainer.py inside fbl
    if "config" not in config_json:
        return config_from_json(config_cls, config_json)
    return config_from_json(config_cls, config_json["config"])


def train_model_distributed(config):
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
    context.obj = {}

    def load_config():
        if "config_json" not in context.obj:
            if config_file:
                with open(config_file) as file:
                    config = json.load(file)
            elif config_json:
                config = json.loads(config_json)
            else:
                click.echo("No config file specified, reading from stdin")
                config = json.load(sys.stdin)
        # Cache the config object so it can be accessed multiple times
        context.obj["config_json"] = config
        return config
    context.obj["load_config"] = load_config


@main.command()
@click.pass_context
def test(context):
    """Test a trained model snapshot."""
    config = parse_config(Mode.TEST, context.obj["load_config"]())
    print("\n=== Starting testing...")
    test_model(config)


@main.command()
@click.pass_context
def train(context):
    """Train a model and save the best snapshot."""
    config = parse_config(Mode.TRAIN, context.obj["load_config"]())
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


if __name__ == "__main__":
    main()
