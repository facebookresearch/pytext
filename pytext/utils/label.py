#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from collections import Counter
from typing import Dict

import torch
from pytext.utils.cuda import tensor
from pytext.utils.typing import WeightingMethod

logging.basicConfig(level=logging.INFO)


def get_custom_or_automatic_label_weights(
    vocab_dict,
    label_counts,
    label_weights_field,
    automatic_label_weighting_method_field,
):
    if label_weights_field and automatic_label_weighting_method_field:
        raise ValueError(
            "Both label_weights and automatic_label_weighting_method are provided. Only one of them is expected"
        )
    if label_weights_field:
        return get_label_weights(vocab_dict, label_weights_field)
    elif automatic_label_weighting_method_field:
        return get_automatic_label_weights(
            vocab_dict, label_counts, automatic_label_weighting_method_field
        )


def get_label_weights(vocab_dict: Dict[str, int], label_weights: Dict[str, float]):
    """
    Get label weights from user-provided label_weights field
    """
    # prune the label_weights to remove the labels that do not exist in the dataset
    pruned_label_weights = {
        vocab_dict[k]: v for (k, v) in label_weights.items() if k in vocab_dict
    }
    if len(pruned_label_weights) != len(label_weights):
        filtered_labels = [k for k in label_weights if k not in vocab_dict]
        logging.warning(
            f"WARNING: these labels are filtered from original label weights \
            {filtered_labels}"
        )
    if len(pruned_label_weights) == 0:
        return None

    # All unspecified classes will get a weight of 1
    weights_tensor = [1] * len(vocab_dict)
    for k, v in pruned_label_weights.items():
        weights_tensor[k] = v
    return tensor(weights_tensor, dtype=torch.float)


def get_automatic_label_weights(
    vocab_dict, label_counts, automatic_label_weighting_method
):
    """
    This function contains the logic on which automatic label weighting method to use based on the config. Label weights are automatically calculated from all training examples.
    Due to the fact that the label distribution can be highly skewed resulting in excessively weights, we implemented two normalization method, 'sqrt' and 'cap', to normalize the weights.
    """
    if automatic_label_weighting_method == WeightingMethod.CLASS_RATIO:
        automatic_label_weights = get_auto_label_weights(vocab_dict, label_counts)
    elif automatic_label_weighting_method == WeightingMethod.SQRT_RATIO:
        automatic_label_weights = get_normalized_sqrt_label_weights(
            vocab_dict, label_counts
        )
    elif automatic_label_weighting_method == WeightingMethod.CAPPED_RATIO:
        automatic_label_weights = get_normalized_cap_label_weights(
            vocab_dict, label_counts
        )
    else:
        raise ValueError(
            f"ERROR: weighting method {automatic_label_weighting_method} not recognized."
        )

    logging.info(
        f"automatic_label_weighting_method is {automatic_label_weighting_method} and label_weights is {automatic_label_weights}"
    )
    return automatic_label_weights


def get_auto_label_weights(vocab_dict: Dict[str, int], label_counts: Counter):
    """
    label weights automatically calculated from training data
    """
    label_weights = {}
    for label in vocab_dict:
        if label in label_counts:
            pos_count = label_counts[label]
            neg_count = sum(label_counts.values()) - pos_count
            weight = neg_count / pos_count
            label_weights[label] = weight
        else:
            label_weights[label] = 1.0

    # initialize
    weights_tensor = torch.ones(1, len(label_weights.keys()))
    for label, weight in label_weights.items():
        weights_tensor[0][vocab_dict[label]] = weight

    return weights_tensor


def get_normalized_sqrt_label_weights(
    vocab_dict: Dict[str, int], label_counts: Counter
):
    """
    label weights automatically calculated from training data, normalized by sqrt
    """
    auto_weights_tensor = get_auto_label_weights(vocab_dict, label_counts)
    return torch.sqrt(auto_weights_tensor)


def get_normalized_cap_label_weights(vocab_dict: Dict[str, int], label_counts: Counter):
    """
    label weights automatically calculated from training data, normalized and capped by 1
    """
    normalized_label_weights = {}
    avg_label_count = sum(label_counts.values()) / len(label_counts.keys())
    for label in vocab_dict:
        if label in label_counts:
            count = label_counts[label]
            if count > avg_label_count:
                label_weight = avg_label_count / count
                normalized_label_weights[label] = label_weight
            else:
                normalized_label_weights[label] = 1.0
        else:
            normalized_label_weights[label] = 1.0

    # initialize
    cap_label_weights_tensor = torch.ones(1, len(vocab_dict.keys()))
    for label, index in vocab_dict.items():
        cap_label_weights_tensor[0][index] = normalized_label_weights[label]

    return cap_label_weights_tensor
