#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from contextlib import contextmanager

from . import cuda


FP16_ENABLED = False
DELAY_UNSCALE = False


@contextmanager
def delay_unscale():
    global DELAY_UNSCALE

    # delay_unscale is required for gradients accumulation, model accumulate
    # gradient on FP16 parameters when set to True and using the same loss_scale
    old_delay_unscale = DELAY_UNSCALE
    DELAY_UNSCALE = True
    try:
        yield
    finally:
        DELAY_UNSCALE = old_delay_unscale


def set_fp16(fp16_enabled: bool):
    global FP16_ENABLED

    if fp16_enabled:
        if not cuda.CUDA_ENABLED:
            raise RuntimeError("Cuda is not available, should not running fp16...")

        FP16_ENABLED = fp16_enabled


def maybe_float(tensor):
    if FP16_ENABLED and tensor.type().split(".")[-1] == "HalfTensor":
        return tensor.float()
    else:
        return tensor


def maybe_half(tensor):
    if FP16_ENABLED and tensor.type().split(".")[-1] == "FloatTensor":
        return tensor.half()
    else:
        return tensor


def pad_length(n):
    if FP16_ENABLED:
        # To take advantage of tensor core, length should be multiple of 8
        remainder = n % 8
        if remainder > 0:
            n = n + 8 - remainder

    return n
