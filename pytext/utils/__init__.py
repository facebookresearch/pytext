#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


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
