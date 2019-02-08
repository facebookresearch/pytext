#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from pytext.models.module import create_module

from .bilstm_doc_attention import BiLSTMDocAttention
from .docnn import DocNNRepresentation
from .representation_base import RepresentationBase


class QueryDocumentPairwiseRankingRep(RepresentationBase):
    """Wrapper representation for a query with 2 reponses.
        where model is trained on pairwise ranking loss
        first input: pos_response (higher ranked)
        second input: neg_response (lower ranked)
        third input: query
        both responses and the query use the same embeddings
    """

    class Config(RepresentationBase.Config):
        subrepresentation: Union[
            BiLSTMDocAttention.Config, DocNNRepresentation.Config
        ] = BiLSTMDocAttention.Config()
        # should the query and the response share representations?
        shared_representations: bool = True

    def __init__(self, config: Config, embed_dim: Tuple[int, ...]) -> None:
        super().__init__(config)
        num_subrepresentations = 2
        assert len(embed_dim) == 1
        # TODO: allow query and response embed_dims to be different
        if config.shared_representations:
            self.subrepresentations = nn.ModuleList(
                itertools.repeat(
                    create_module(config.subrepresentation, embed_dim=embed_dim[0]),
                    num_subrepresentations,
                )
            )
        else:
            self.subrepresentations = nn.ModuleList(
                create_module(config.subrepresentation, embed_dim=embed_dim[0])
                for x in range(num_subrepresentations)
            )

        self.representation_dim = self.subrepresentations[0].representation_dim
        self.representation_dim = (num_subrepresentations, self.representation_dim)

    # Takes care of dropping the extra return value of LSTM-based rep's (state).
    @staticmethod
    def _represent(
        rep: RepresentationBase, embs: torch.Tensor, lens: torch.Tensor
    ) -> torch.Tensor:
        representation = rep(embs, lens)
        if isinstance(representation, tuple):
            return representation[0]
        return representation

    def forward(
        self, embeddings: Tuple[torch.Tensor, ...], *lengths: torch.Tensor
    ) -> List[torch.Tensor]:
        # The leftmost inputs already come sorted by length. The others need to
        # be sorted as well, for packing. We do it manually.
        # input: pos_embed, neg_embed, query_embed
        # pos_embed and neg_embed use the same representation (LSTM)

        sorted_inputs = [(embeddings[0], lengths[0])]
        sorted_indices = [None]
        for embs, lens in zip(embeddings[1:], lengths[1:]):
            lens_sorted, sorted_idx = lens.sort(descending=True)
            embs_sorted = embs[sorted_idx]
            sorted_inputs.append((embs_sorted, lens_sorted))
            sorted_indices.append(sorted_idx)

        # the first and second embeddings are run through the same LSTM
        representations = [
            self._represent(self.subrepresentations[0], embs, lens)
            for (embs, lens) in sorted_inputs[0:2]
        ]
        representations.append(
            self._represent(
                self.subrepresentations[1], sorted_inputs[2][0], sorted_inputs[2][1]
            )
        )

        # Put the inputs back in the original order, so they still match up to
        # each other as well as the targets.
        unsorted_representations = [representations[0]]
        for sorted_idx, rep in zip(sorted_indices[1:], representations[1:]):
            _, unsorted_idx = sorted_idx.sort()
            unsorted_representations.append(rep[unsorted_idx])

        return unsorted_representations
