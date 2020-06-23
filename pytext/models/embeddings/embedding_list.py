#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Iterable, List, Tuple, Union

import torch
import torch.nn as nn
from pytext.utils.usage import log_class_usage
from torch.nn import ModuleList
from torch.utils.tensorboard import SummaryWriter

from .embedding_base import EmbeddingBase


class EmbeddingList(EmbeddingBase, ModuleList):
    """
    There are more than one way to embed a token and this module provides a way
    to generate a list of sub-embeddings, concat embedding tensors into a single
    Tensor or return a tuple of Tensors that can be used by downstream modules.

    Args:
        embeddings (Iterable[EmbeddingBase]): A sequence of embedding modules to
        embed a token.
        concat (bool): Whether to concatenate the embedding vectors emitted from
        `embeddings` modules.

    Attributes:
        num_emb_modules (int): Number of flattened embeddings in `embeddings`,
            e.g: ((e1, e2), e3) has 3 in total
        input_start_indices (List[int]): List of indices of the sub-embeddings
            in the embedding list.
        concat (bool): Whether to concatenate the embedding vectors emitted from
            `embeddings` modules.
        embedding_dim: Total embedding size, can be a single int or tuple of
            int depending on concat setting
    """

    def __init__(self, embeddings: Iterable[EmbeddingBase], concat: bool) -> None:
        EmbeddingBase.__init__(self, 0)
        embeddings = list(filter(None, embeddings))
        self.num_emb_modules = sum(emb.num_emb_modules for emb in embeddings)
        embeddings_list, input_start_indices = [], []
        start = 0
        for emb in embeddings:
            if emb.embedding_dim > 0:
                embeddings_list.append(emb)
                input_start_indices.append(start)
            start += emb.num_emb_modules
        ModuleList.__init__(self, embeddings_list)
        self.input_start_indices = input_start_indices
        self.concat = concat
        assert len(self) > 0, "must have at least 1 sub embedding"
        embedding_dims = tuple(emb.embedding_dim for emb in self)
        self.embedding_dim = sum(embedding_dims) if concat else embedding_dims
        log_class_usage(__class__)

    def forward(self, *emb_input) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Get embeddings from all sub-embeddings and either concatenate them
        into one Tensor or return them in a tuple.

        Args:
            *emb_input (type): Sequence of token level embeddings to combine.
                The inputs should match the size of configured embeddings. Each
                of them is either a Tensor or a tuple of Tensors.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: If `concat` is True then
                a Tensor is returned by concatenating all embeddings. Otherwise
                all embeddings are returned in a tuple.

        """
        # tokens dim: (bsz, max_seq_len) -> (bsz, max_seq_len, dim) OR
        # (bsz, max_num_sen, max_seq_len) -> (bsz, max_num_sen, max_seq_len, dim)
        # for seqnn
        if self.num_emb_modules != len(emb_input):
            raise Exception(
                f"expecting {self.num_emb_modules} embeddings, "
                + f"but got {len(emb_input)} input"
            )
        tensors = []
        for emb, start in zip(self, self.input_start_indices):
            end = start + emb.num_emb_modules
            input = emb_input[start:end]
            # single embedding
            if len(input) == 1:
                # the input for the single embedding is a tuple or list of tensors
                if isinstance(input[0], list) or isinstance(input[0], tuple):
                    [input] = input
            emb_tensor = emb(*input)
            tensors.append(emb_tensor)

        if self.concat:
            return torch.cat(tensors, -1)
        else:
            return tuple(tensors) if len(tensors) > 1 else tensors[0]

    def get_param_groups_for_optimizer(self) -> List[Dict[str, nn.Parameter]]:
        """
        Organize child embedding parameters into param_groups (or layers), so the
        optimizer and / or schedulers can have custom behavior per layer. The
        param_groups from each child embedding are aligned at the first (lowest)
        param_group.
        """
        param_groups: List[Dict[str, nn.Parameter]] = []

        for module_name, embedding_module in self.named_children():
            child_params = embedding_module.get_param_groups_for_optimizer()

            for i, child_param_group in enumerate(child_params):
                if i >= len(param_groups):
                    param_groups.append({})

                for param_name, param in child_param_group.items():
                    param_name = "%s.%s" % (module_name, param_name)
                    param_groups[i][param_name] = param

        return param_groups

    def visualize(self, summary_writer: SummaryWriter):
        for child in self:
            child.visualize(summary_writer)
