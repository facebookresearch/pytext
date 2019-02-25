#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .pytext_config import LATEST_VERSION


def v0_to_v1(json_config):
    # migrate optimizer params
    [task] = json_config["task"].values()
    if (
        "optimizer" not in task
        or "Adam" in task["optimizer"]
        or "SGD" in task["optimizer"]
        or "NAG" in task["optimizer"]
    ):
        return json_config
    op_type = task["optimizer"].get("type", "adam")
    if op_type == "adam":
        op_config = {"Adam": {}}
        for key in ["lr", "weight_decay"]:
            if key in task["optimizer"]:
                op_config["Adam"][key] = task["optimizer"][key]
    elif op_type == "sgd":
        op_config = {"SGD": {}}
        for key in ["lr", "momentum"]:
            if key in task["optimizer"]:
                op_config["SGD"][key] = task["optimizer"][key]
    elif op_type == "nag":
        op_config = {"NAG": {}}
        for key in ["lr", "weight_decay", "momentum"]:
            if key in task["optimizer"]:
                op_config["NAG"][key] = task["optimizer"][key]
    else:
        raise ValueError("Migration not supported for your optimizer")
    task["optimizer"] = op_config
    json_config["version"] = 1
    return json_config


adapters = {0: v0_to_v1}


def upgrade_one_version(json_config):
    current_version = json_config.get("version", 0)
    adapter = adapters.get(current_version)
    if not adapter:
        raise Exception(f"no adapter found for version {current_version}")
    json_config = adapter(json_config)
    json_config["version"] = current_version + 1
    return json_config


def upgrade_to_latest(json_config):
    current_version = json_config.get("version", 0)
    if current_version > LATEST_VERSION:
        raise Exception(
            f"config version {json_config['version']} shouldn't exceed lastest \
            version {LATEST_VERSION}"
        )
    while current_version != LATEST_VERSION:
        json_config = upgrade_one_version(json_config)
        current_version = json_config["version"]
    return json_config
