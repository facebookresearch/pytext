#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

_APEX_DISABLED = False
try:
    from apex import amp
except ImportError:
    print("Install apex from https://github.com/NVIDIA/apex/.")
    _APEX_DISABLED = True
except AttributeError as e:
    print(f"Fail to import apex: {e}")
    _APEX_DISABLED = True

from . import cuda


"""
Tips:
1. Must run fp16 on latest generation (Volta V100) GPU, CUDA 9.1 or newer
2. Additionally:
    - Batch size should be a multiple of 8
    - Tokens size should be a multiple of 8
    - Embedding layers should be padded to be a multiple of 8
    - Ideally, everything should be a multiple of 8 (e.g padding, etc)
3. Larger batch_size could increase GPU utilization and better performance.
4. Amp might not work well for model that require too many back-and-forth
    parameter casting between fp16 and fp32.
"""

"""
Apex amp: https://github.com/NVIDIA/apex/tree/master/apex/amp

FP32 Master Weights <--(step)-- FP32 Gradients <--(unscale)-- Scaled FP16 Gradients
       |                                                        |
(copy) |                                                        | (backprop)
       |                                                        |
FP16 Weights --(forward)--> FP32 Loss --(loss scaling)--> Scaled FP32 Loss

For Apex.amp, it handle the Mixed precision training in the folloing ways
1. [Master weights]: master weights(e.g fp32) will mainted by PyTorch model
2. [Forward & Backward]: amp wrap PyTorch functions, it will cast inputs &
   weights into fp16 or fp32, _amp_handle caches the casted arguments.
3. [Loss scaling]: _amp_handle handle loss scaling and unscaling

Using amp require adding three lines of code.
1. _amp_handle = amp.init(enabled=fp16_enabled)
2. optimizer = _amp_handle.wrap_optimizer(optimizer)
3. with optimizer.scale_loss(loss) as scaled_loss: scaled_loss.backward()
"""


_FP16_ENABLED = False
_amp_handle = None


def set_fp16(fp16_enabled: bool):
    global _FP16_ENABLED
    global _amp_handle

    if _APEX_DISABLED:
        return

    _FP16_ENABLED = fp16_enabled
    if _FP16_ENABLED:
        if not cuda.CUDA_ENABLED:
            raise RuntimeError("Cuda is not available, should not running fp16...")

        _amp_handle = amp.init(enabled=fp16_enabled)


def wrap_optimizer(optimizer):
    if _FP16_ENABLED:
        return _amp_handle.wrap_optimizer(optimizer)
    else:
        return optimizer


def backward(optimizer, loss):
    if _FP16_ENABLED:
        # 1. Use automatic loss scaling to best use fp16 range (skip step if overflow)
        # 2. Clear handle's cache of casted parameters before the next optimizer step
        with optimizer.scale_loss(loss) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()


def deactivate():
    global _FP16_ENABLED

    if _FP16_ENABLED:
        # restoring uncasted versions of functions
        _amp_handle._deactivate()
        _FP16_ENABLED = False
    else:
        pass


def maybe_float(tensor):
    if _FP16_ENABLED and tensor.type().split(".")[-1] == "HalfTensor":
        return tensor.float()
    else:
        return tensor


def maybe_half(tensor):
    if _FP16_ENABLED and tensor.type().split(".")[-1] == "FloatTensor":
        return tensor.half()
    else:
        return tensor


def pad_length(n):
    if _FP16_ENABLED:
        # To take advantage of tensor core, length should be multiple of 8
        remainder = n % 8
        if remainder > 0:
            n = n + 8 - remainder

    return n
