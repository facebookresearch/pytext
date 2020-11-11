#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from collections import Sequence


def cls_vars(cls):
    return [v for n, v in vars(cls).items() if not n.startswith("_")]


def set_random_seeds(seed, use_deterministic_cudnn):
    import random

    import numpy as np
    import torch
    from pytext.utils import cuda

    # See https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda.CUDA_ENABLED and use_deterministic_cudnn:
        print(
            """WARNING: Your training might be slower because you have set
        use_deterministic_cudnn flag to True. Read
        https://pytorch.org/docs/stable/notes/randomness.html and
        https://discuss.pytorch.org/t/what-is-the-differenc-between-cudnn-deterministic-and-cudnn-benchmark/38054
        """
        )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def recursive_map(seq, func):
    """This is similar to the build-in map function but works for nested lists.
    Useful for transforming tensors serialized with .tolist()
    """
    for item in seq:
        if isinstance(item, Sequence):
            yield type(item)(recursive_map(item, func))
        else:
            yield func(item)


def round_seq(seq, ndigits):
    """Rounds a nested sequence of floats to ndigits precision.
    Useful for rounding tensors serialized with .tolist()
    """
    return type(seq)(recursive_map(seq, lambda item: round(item, ndigits)))
