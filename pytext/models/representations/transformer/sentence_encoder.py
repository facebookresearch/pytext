#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
from typing import List, Optional

import torch
from pytext.utils.usage import log_class_usage
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
        log_class_usage(__class__)

    def forward(self, tokens):
        all_layers = self.extract_features(tokens)  # list of [T x B x C]
        return [layer.transpose(0, 1) for layer in all_layers]

    def extract_features(self, tokens):
        # support passing in a single sentence
        torch._assert(
            tokens.dim() == 1 or tokens.dim() == 2, "tokens should be a 1D or 2D tensor"
        )
        tokens = tokens.view(-1, tokens.shape[-1])
        return self.transformer(tokens)

    def load_roberta_state_dict(self, state_dict):
        return self.load_state_dict(translate_roberta_state_dict(state_dict))


class PostEncoder(SentenceEncoder):
    def forward(self, tokens: torch.Tensor, dense: List[torch.Tensor]):
        all_layers = self.extract_features(tokens, dense)  # list of [T x B x C]
        return [layer.transpose(0, 1) for layer in all_layers]

    def extract_features(self, tokens: torch.Tensor, dense: List[torch.Tensor]):
        # support passing in a single sentence
        tokens = tokens.view(-1, tokens.shape[-1])
        return self.transformer(tokens, dense)


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


def check_state_keys(state, keys_regex):
    """check if keys exists in state using full python paths"""
    regex = re.compile(keys_regex)
    for k, v in state.items():
        if regex.findall(k):
            return True
    return False


def merge_input_projection(state):
    """
    New checkpoints of fairseq multihead attention split in_projections into
    k,v,q projections. This function merge them back to to make it compatible.
    """
    items_to_add = {}
    keys_to_remove = []
    bias_suffix = ["q_proj.bias", "k_proj.bias", "v_proj.bias"]
    weight_suffix = ["q_proj.weight", "k_proj.weight", "v_proj.weight"]

    def override_state(k, suffix, new_suffix, idx):
        new_key = k[: -len(suffix)] + new_suffix
        dim = state[k].shape[0]
        if new_key not in items_to_add:
            items_to_add[new_key] = (
                torch.zeros_like(state[k]).repeat(3, 1)
                if len(state[k].shape) > 1
                else torch.zeros_like(state[k]).repeat(3)
            )
        items_to_add[new_key][idx * dim : (idx + 1) * dim] = state[k]
        keys_to_remove.append(k)

    for k in state.keys():
        # weights
        for idx, suffix in enumerate(weight_suffix):
            if k.endswith(suffix):
                override_state(k, suffix, "in_proj_weight", idx)
        # bias
        for idx, suffix in enumerate(bias_suffix):
            if k.endswith(suffix):
                override_state(k, suffix, "in_proj_bias", idx)

    for k in keys_to_remove:
        del state[k]

    for key, value in items_to_add.items():
        state[key] = value

    return state


def translate_roberta_state_dict(state_dict):
    """Translate the public RoBERTa weights to ones which match SentenceEncoder."""

    new_state = rename_component_from_root(
        state_dict, "decoder.sentence_encoder", "transformer"
    )
    new_state = rename_component_from_root(
        new_state, "encoder.sentence_encoder", "transformer"
    )

    new_state = rename_state_keys(new_state, "embed_tokens", "token_embedding")
    # TODO: segment_embeddings?
    new_state = rename_state_keys(
        new_state, "embed_positions", "positional_embedding.embedding"
    )
    new_state = rename_state_keys(new_state, "emb_layer_norm", "embedding_layer_norm")
    new_state = rename_state_keys(new_state, "self_attn", "attention")
    # Linformer special handling
    if check_state_keys(new_state, "compress_k"):
        new_state = remove_state_keys(new_state, "compress_layer")
    else:
        new_state = merge_input_projection(new_state)

    new_state = rename_state_keys(new_state, "_proj.(.*)", r"put_projection.\1")
    new_state = rename_state_keys(new_state, "fc1", "residual_mlp.mlp.0")
    new_state = rename_state_keys(new_state, "fc2", "residual_mlp.mlp.3")

    new_state = remove_state_keys(new_state, "^sentence_")
    new_state = remove_state_keys(new_state, "_classification_head.")
    new_state = remove_state_keys(new_state, r"^decoder\.lm_head")
    new_state = remove_state_keys(new_state, r"^encoder\.lm_head")
    new_state = remove_state_keys(new_state, r"segment_embedding")

    return new_state
