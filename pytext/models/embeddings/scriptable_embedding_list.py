#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from inspect import signature
from typing import Dict, Iterable, List, Tuple

import torch
from pytext.models.embeddings import EmbeddingBase
from torch.utils.tensorboard import SummaryWriter


class ScriptableEmbeddingList(EmbeddingBase):
    """

    This class is a Torchscript-friendly version of
    pytext.models.embeddings.EmbeddingList. The main differences are that it
    requires input arguments to be passed in as a list of Tensors, since
    Torchscript does not allow variable arguments, and that it only supports
    concat mode, since Torchscript does not support return value variance.

    """

    # Different embedding classes in PyText take different numbers of input
    # arguments. Therefore, we want to allow users of this class to simply pass
    # in a list of tensors for each embedding. Unfortunately, Torchscript
    # requires that we know the signature of the function we're calling before
    # .forward() is called. These classes and adapter method allow us to
    # provide a flexible interface to callers of this class while satisfying
    # Torchscript.

    # N.B.: it may be possible to generate these classes dynamically, but given
    # that we currently only need two of them it seems better to define them
    # here so we get clean stack traces, autocomplete, etc.

    class Wrapper1(torch.nn.Module):
        def __init__(self, embedding: EmbeddingBase):
            super().__init__()
            self._embedding = embedding

        def forward(self, xs: List[torch.Tensor]):
            return self._embedding(xs[0])

    class Wrapper3(torch.nn.Module):
        def __init__(self, embedding: EmbeddingBase):
            super().__init__()
            self._embedding = embedding

        def forward(self, xs: List[torch.Tensor]):
            return self._embedding(xs[0], xs[1], xs[2])

    @staticmethod
    def _adapt_embedding(embedding: torch.nn.Module) -> torch.nn.Module:
        param_count = len(signature(embedding.forward).parameters)

        if param_count == 1:
            return ScriptableEmbeddingList.Wrapper1(embedding)
        elif param_count == 3:
            return ScriptableEmbeddingList.Wrapper3(embedding)
        raise AssertionError(
            f"Unsupported parameter count {param_count}. If a new embedding "
            "class has been added, you will need to add support in this class."
        )

    def __init__(self, embeddings: Iterable[EmbeddingBase]):
        EmbeddingBase.__init__(self, 0)
        embeddings = list(filter(None, embeddings))
        self.num_emb_modules = sum(emb.num_emb_modules for emb in embeddings)
        embeddings_list: List[EmbeddingBase] = []
        input_start_indices: List[int] = []
        start = 0
        embedding_dim = 0
        for emb in embeddings:
            if emb.embedding_dim > 0:
                embeddings_list.append(emb)
                input_start_indices.append(start)
                embedding_dim += emb.embedding_dim
            start += emb.num_emb_modules
        self.embeddings_list = torch.nn.ModuleList(
            map(ScriptableEmbeddingList._adapt_embedding, embeddings_list)
        )
        self.input_start_indices: Tuple[int] = tuple(input_start_indices)
        assert len(self.embeddings_list) > 0, "must have at least 1 sub embedding"
        self.embedding_dim = embedding_dim

    def forward(self, emb_input: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Get embeddings from all sub-embeddings and either concatenate them
        into one Tensor or return them in a tuple.

        Args:
            emb_input (type): Sequence of token level embeddings to combine.
                The inputs should match the size of configured embeddings. Each
                of them is a List of Tensors.

        Returns:
            torch.Tensor: a Tensor is returned by concatenating all embeddings.

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
        for emb, start in zip(self.embeddings_list, self.input_start_indices):
            tensors.append(emb(emb_input[start]))

        return torch.cat(tensors, 2)

    def get_param_groups_for_optimizer(self) -> List[Dict[str, torch.nn.Parameter]]:
        """
        Organize child embedding parameters into param_groups (or layers), so the
        optimizer and / or schedulers can have custom behavior per layer. The
        param_groups from each child embedding are aligned at the first (lowest)
        param_group.
        """
        param_groups: List[Dict[str, torch.nn.Parameter]] = []

        for module_name, embedding_module in self.embeddings_list.named_children():
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
