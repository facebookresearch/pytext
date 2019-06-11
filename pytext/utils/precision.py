#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from sys import stderr

import torch

from . import cuda


_APEX_DISABLED = False
try:
    from apex import amp, fp16_utils

    class FP16_Optimizer(fp16_utils.FP16_Optimizer):
        def finalize(self) -> bool:
            return self.optimizer.finalize()


except ImportError:
    print("Install apex from https://github.com/NVIDIA/apex/.", file=stderr)
    _APEX_DISABLED = True
except AttributeError as e:
    print(f"Fail to import apex: {e}")
    _APEX_DISABLED = True


"""
Tips:
1. Recommand run fp16 on latest generation (Volta V100) GPU, CUDA 9.1 or newer
   to leverage tensor cores, which provide 8x more throughput than single
   precision math pipelines.
2. Additionally:
    - Batch size should be a multiple of 8
    - Tokens size should be a multiple of 8
    - Embedding layers should be padded to be a multiple of 8
    - Ideally, everything should be a multiple of 8 (e.g padding, etc)
3. Larger batch_size might increase GPU utilization and better performance.
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

Using amp require adding three lines of code.
https://nvidia.github.io/apex/amp.html
1. Allow Amp to perform casts as required by the opt_level:
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

2. loss.backward() replace with:
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()

3. torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) replace with:
torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)

Opt level explaination (from Nvidia Apex):

* O1:  Insert automatic casts around Pytorch functions and Tensor methods
 - The type of your model's weights is not altered.  However, internally,
   Pytorch functions are patched to cast any Tensor Core-friendly ops to FP16
   for speed, while operations that might benefit from the additional stability
   of FP32 are patched to cast their inputs to fp32.
 - O1 is the safest way to try mixed precision training, and is recommended when
   trying mixed precision training for the first time.

* O2:  FP16 training with FP32 batchnorm and FP32 master weights.
 - Calls .half() on your model, converting the entire model (except for batchnorms)
   to FP16.  Batchnorms are retained in FP32 for additional stability.
 - The forward pass is patched to cast incoming Tensors to FP16, so you don't
   need to change your data pipeline.
 - O2 creates FP32 master weights outside the model and patches any optimizers
   to update these master weights, then copy the master weights into the FP16
   model weights.
"""


_FP16_ENABLED = False
_OPT_LEVEL = None


def set_fp16(fp16_enabled: bool):
    global _FP16_ENABLED

    if _APEX_DISABLED:
        return

    if fp16_enabled:
        if not cuda.CUDA_ENABLED:
            raise RuntimeError("Cuda is not available, should not running fp16...")

        _FP16_ENABLED = fp16_enabled


def initialize(model, optimizer):
    global _OPT_LEVEL

    if _FP16_ENABLED:
        _OPT_LEVEL = "O2" if model.SUPPORT_FP16_OPTIMIZER else "O1"
        return amp.initialize(model, optimizer, opt_level=_OPT_LEVEL)
    else:
        return model, optimizer


def backward(optimizer, loss):
    if _FP16_ENABLED:
        # 1. Use automatic loss scaling to best use fp16 range
        # 2. Clear handle's cache of casted parameters
        if loss > 0:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
    else:
        loss.backward()


def clip_grad_norm(model, optimizer, max_clip_norm):
    if _FP16_ENABLED:
        # Refer: https://nvidia.github.io/apex/advanced.html
        return torch.nn.utils.clip_grad_norm_(
            amp.master_params(optimizer), max_clip_norm
        )
    else:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_clip_norm)


def deactivate(model):
    # Warning: this function is expected to be called after train finished.
    # In case need to deactivate before train, should invoke unwrap_optimizer first.

    global _FP16_ENABLED
    global _OPT_LEVEL

    if _FP16_ENABLED:
        if _OPT_LEVEL == "O2":
            # convert model parameters back to fp32
            model.float()
            _OPT_LEVEL = None
        else:
            # restoring uncasted versions of functions
            amp._amp_state.handle._deactivate()
        _FP16_ENABLED = False


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
