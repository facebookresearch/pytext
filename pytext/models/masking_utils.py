#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum

import numpy as np
import torch


class MaskingStrategy(Enum):
    RANDOM = "random"
    FREQUENCY = "frequency_based"


def random_masking(tokens: torch.tensor, mask_prob: float) -> torch.Tensor:
    """
    Function to mask tokens randomly.

    Inputs:
        1) tokens: Tensor with token ids of shape (batch_size x seq_len)
        2) mask_prob: Probability of masking a particular token

    Outputs:
        mask: Tensor with same shape as input tokens (batch_size x seq_len)
            with masked  tokens represented by a 1 and everything else as 0.
    """
    batch_size, seq_len = tokens.size()
    num_masked_per_seq = int(seq_len * mask_prob)

    mask = np.zeros((batch_size, seq_len), dtype=np.int_)
    mask[:, :num_masked_per_seq] = 1
    for row in mask:
        np.random.shuffle(row)
    mask = torch.from_numpy(mask).to(tokens.device)
    return mask


def frequency_based_masking(
    tokens: torch.tensor, token_sampling_weights: np.ndarray, mask_prob: float
) -> torch.Tensor:
    """
    Function to mask tokens based on frequency.

    Inputs:
        1) tokens: Tensor with token ids of shape (batch_size x seq_len)
        2) token_sampling_weights: numpy array with shape (batch_size x seq_len)
            and each element representing the sampling weight assicated with
            the corresponding token in tokens
        3) mask_prob: Probability of masking a particular token

    Outputs:
        mask: Tensor with same shape as input tokens (batch_size x seq_len)
            with masked  tokens represented by a 1 and everything else as 0.
    """
    batch_size, seq_len = tokens.size()
    num_masked_per_batch = int(batch_size * seq_len * mask_prob)

    indices = tokens.cpu().numpy().flatten()

    # get the weights associated with each token
    weights = np.take(token_sampling_weights, indices)

    # sample tokens based on the computed weights
    tokens_to_mask = np.random.choice(
        len(weights), num_masked_per_batch, replace=False, p=weights / weights.sum()
    )
    mask = torch.zeros(batch_size * seq_len)
    mask[tokens_to_mask] = 1
    mask = mask.view(batch_size, seq_len).long().to(tokens.device)
    return mask
