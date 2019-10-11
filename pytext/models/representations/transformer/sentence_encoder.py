#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
from typing import Optional

from torch import nn

from .transformer import Transformer


class SentenceEncoder(nn.Module):
    """
    This is a TorchScriptable implementation of RoBERTa from fairseq
    for the purposes of creating a productionized RoBERTa model. It distills just
    the elements which are required to implement the RoBERTa model, and within that
    is restructured and rewritten to be able to be compiled by TorchScript for
    production use cases.

    This SentenceEncoder can load in the public RoBERTa weights directly with
    `load_roberta_state_dict`, which will translate the keys as they exist in
    the publicly released RoBERTa to the correct structure for this implementation.
    The default constructor value will have the same size and shape as that model.

    To use RoBERTa with this, download the RoBERTa public weights as `roberta.weights`

    >>> encoder = SentenceEncoder()
    >>> weights = torch.load("roberta.weights")
    >>> encoder.load_roberta_state_dict(weights)

    Within this you will still need to preprocess inputs using fairseq and the publicly
    released vocabs, and finally place this encoder in a model alongside say an MLP
    output layer to do classification.
    """

    def __init__(self, transformer: Optional[Transformer] = None):
        super().__init__()
        self.transformer = transformer or Transformer()

    def forward(self, tokens):
        all_layers = self.extract_features(tokens)
        last_layer = all_layers[-1]  # T x B x C
        return last_layer.transpose(0, 1)

    def extract_features(self, tokens):
        # support passing in a single sentence
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        return self.transformer(tokens)

    def load_roberta_state_dict(self, state_dict):
        return self.load_state_dict(translate_roberta_state_dict(state_dict))


def remove_state_keys(state, keys_regex):
    """Remove keys from state that match a regex"""
    regex = re.compile(keys_regex)
    return {k: v for k, v in state.items() if not regex.findall(k)}


def rename_state_keys(state, keys_regex, replacement):
    """Rename keys from state that match a regex; replacement can use capture groups"""
    regex = re.compile(keys_regex)
    return {
        (k if not regex.findall(k) else regex.sub(replacement, k)): v
        for k, v in state.items()
    }


def rename_component_from_root(state, old_name, new_name):
    """Rename keys from state using full python paths"""
    return rename_state_keys(
        state, "^" + old_name.replace(".", r"\.") + ".?(.*)$", new_name + r".\1"
    )


def translate_roberta_state_dict(state_dict):
    """Translate the public RoBERTa weights to ones which match SentenceEncoder."""
    new_state = rename_component_from_root(
        state_dict, "decoder.sentence_encoder", "transformer"
    )
    new_state = rename_state_keys(new_state, "embed_tokens", "token_embedding")
    # TODO: segment_embeddings?
    new_state = rename_state_keys(
        new_state, "embed_positions", "positional_embedding.embedding"
    )
    new_state = rename_state_keys(new_state, "emb_layer_norm", "embedding_layer_norm")
    new_state = rename_state_keys(new_state, "self_attn", "attention")
    new_state = rename_state_keys(new_state, "_proj.(.*)", r"put_projection.\1")
    new_state = rename_state_keys(new_state, "fc1", "residual_mlp.mlp.0")
    new_state = rename_state_keys(new_state, "fc2", "residual_mlp.mlp.3")

    new_state = remove_state_keys(new_state, "^sentence_")
    new_state = remove_state_keys(new_state, r"^decoder\.lm_head")
    new_state = remove_state_keys(new_state, r"segment_embedding")
    return new_state
