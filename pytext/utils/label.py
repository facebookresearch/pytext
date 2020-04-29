#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict

import torch
from pytext.utils.cuda import tensor


def get_label_weights(vocab_dict: Dict[str, int], label_weights: Dict[str, float]):
    # prune the label_weights to remove the labels that do not exist in the dataset
    pruned_label_weights = {
        vocab_dict[k]: v for (k, v) in label_weights.items() if k in vocab_dict
    }
    if len(pruned_label_weights) != len(label_weights):
        filtered_labels = [k for k in label_weights if k not in vocab_dict]
        print(
            f"Warning: these labels are filtered from original label weights \
            {filtered_labels}"
        )
    if len(pruned_label_weights) == 0:
        return None

    # All unspecified classes will get a weight of 1
    weights_tensor = [1] * len(vocab_dict)
    for k, v in pruned_label_weights.items():
        weights_tensor[k] = v
    return tensor(weights_tensor, dtype=torch.float)
