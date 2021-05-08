#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy

from fairseq.data.encoders.gpt2_bpe_utils import Encoder as GPT2BPEEncoder


class PickleableGPT2BPEEncoder(GPT2BPEEncoder):
    """Fairseq's encoder stores the regex module as a local reference on its encoders,
    which means they can't be saved via pickle.dumps or torch.save. This modified
    their save/load logic doesn't store the module, and restores the reference
    after re-inflating."""

    def __getstate__(self):
        # make a shallow copy of state to avoid side effect on the original object
        state = copy.copy(vars(self))
        state.pop("re")
        return state

    def __setstate__(self, state):
        vars(self).update(state)
        import regex

        self.re = regex
