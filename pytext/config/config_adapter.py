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


def v1_to_v2(json_config):
    # migrate optimizer params
    [task] = json_config["task"].values()
    if (
        "scheduler" not in task
        or task["scheduler"] is None
        or "LmFineTuning" in task["scheduler"]
        or "StepLR" in task["scheduler"]
        or "ReduceLROnPlateau" in task["scheduler"]
        or "CosineAnnealingLR" in task["scheduler"]
        or "ExponentialLR" in task["scheduler"]
    ):
        return json_config
    op_type = task["scheduler"].get("type")
    assert op_type is not None
    if op_type == "step_lr":
        op_config = {"StepLR": {}}
        for key in ["step_size", "gamma"]:
            if key in task["scheduler"]:
                op_config["StepLR"][key] = task["scheduler"][key]
    elif op_type == "lm_fine_tuning":
        op_config = {"LmFineTuning": {}}
        for key in [
            "cut_frac",
            "ratio",
            "non_pretrained_param_groups",
            "lm_lr_multiplier",
            "lm_use_per_layer_lr",
            "lm_gradual_unfreezing",
            "last_epoch",
        ]:
            if key in task["scheduler"]:
                op_config["LmFineTuning"][key] = task["scheduler"][key]
    elif op_type == "reduce_lr_on_plateau":
        op_config = {"ReduceLROnPlateau": {}}
        for key in [
            "lower_is_better",
            "factor",
            "patience",
            "min_lr",
            "threshold",
            "threshold_is_absolute",
            "cooldown",
        ]:
            if key in task["scheduler"]:
                op_config["ReduceLROnPlateau"][key] = task["scheduler"][key]
    elif op_type == "cosine_annealing_lr":
        op_config = {"CosineAnnealingLR": {}}
        for key in ["t_max", "eta_min"]:
            if key in task["scheduler"]:
                op_config["CosineAnnealingLR"][key] = task["scheduler"][key]
    elif op_type == "exponential_lr":
        op_config = {"ExponentialLR": {}}
        for key in ["gamma"]:
            if key in task["scheduler"]:
                op_config["ExponentialLR"][key] = task["scheduler"][key]
    else:
        raise ValueError(
            "Migration for your scheduler not supported. Please add it here."
        )
    task["scheduler"] = op_config
    json_config["version"] = 2
    return json_config


adapters = {0: v0_to_v1, 1: v1_to_v2}


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
