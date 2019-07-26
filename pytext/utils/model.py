#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Iterable, Optional

import torch

from .cuda import Variable, zerovar


def to_onehot(feat: Variable, size: int) -> Variable:
    """
    Transform features into one-hot vectors
    """
    dim = [d for d in feat.size()]
    vec_ = torch.unsqueeze(feat, len(dim))
    dim.append(size)
    one_hot = zerovar(dim)
    one_hot.data.scatter_(len(dim) - 1, vec_.data, 1)
    return one_hot


def get_mismatched_param(
    models: Iterable[torch.nn.Module],
    rel_epsilon: Optional[float] = None,
    abs_epsilon: Optional[float] = None,
) -> str:
    """
    Return the name of the first mismatched parameter.
    Return an empty string if all the parameters of the modules are identical.
    """
    if rel_epsilon is None and abs_epsilon is not None:
        print("WARNING: rel_epsilon is not specified, abs_epsilon is ignored.")

    if len(models) <= 1:
        return True

    # Verify all models have the same params.
    for model in models[1:]:
        for name, param in models[0].state_dict().items():
            param_here = model.state_dict()[name]

            # If epsilon is specified, do approx comparison.
            if rel_epsilon is not None:
                if abs_epsilon is not None:
                    if not torch.allclose(
                        param, param_here, rtol=rel_epsilon, atol=abs_epsilon
                    ):
                        return name
                else:
                    if not torch.allclose(param, param_here, rtol=rel_epsilon):
                        return name
            else:
                if not torch.equal(param, param_here):
                    return name
    return ""
