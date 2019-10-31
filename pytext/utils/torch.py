#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from pytext.torchscript.tensorizer import VectorNormalizer  # noqa
from pytext.torchscript.tokenizer import ScriptBPE as BPE  # noqa
from pytext.torchscript.utils import (  # noqa
    add_bos_eos_2d,
    add_special_token_2d,
    list_max,
    list_membership,
    long_tensor_2d,
    make_byte_inputs,
    make_sequence_lengths,
    pad_2d,
    pad_2d_mask,
    reverse_tensor_list,
    utf8_chars,
)
from pytext.torchscript.vocab import ScriptVocabulary as Vocabulary  # noqa
from pytext.utils import cuda


# Note: this file is used to load the training checkpoint (backward compatibility)
# For any new usecase, please import directly from pytext.torchscript.


class CPUOnlyParameter(torch.nn.Parameter):
    def __init__(self, *args, **kwargs):
        assert (
            cuda.DISTRIBUTED_WORLD_SIZE <= 1
        ), "Multiple GPUs not supported for cpu_only embeddings"
        super().__init__(*args, **kwargs)

    def cuda(self, device=None):
        # We do nothing because this Parameter should only be on the CPU
        return self
