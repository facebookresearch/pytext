#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from pytext.models.module import create_module

from .bilstm_doc_attention import BiLSTMDocAttention
from .docnn import DocNNRepresentation
from .representation_base import RepresentationBase


SubRepresentation = Union[BiLSTMDocAttention.Config, DocNNRepresentation.Config]


class PairRepresentation(RepresentationBase):
    """Wrapper representation for a pair of inputs.

    Takes a tuple of inputs: the left sentence, and the right sentence(s). Returns
    a representation of the pair of sentences, either as a concatenation of the two
    sentence embeddings or as a "siamese" representation which also includes their
    difference and elementwise product (arXiv:1705.02364).
    If more than two inputs are provided, the extra inputs are assumed to be extra
    "right" sentences, and the output will be the stacked pair representations
    of the left sentence together with all right sentences. This is more efficient
    than separately computing all these pair representations, because the left
    sentence will not need to be re-embedded multiple times.
    """

    class Config(RepresentationBase.Config):
        """
        Attributes:
            encode_relations (bool): if `false`, return the concatenation of the two
                representations; if `true`, also concatenate their pairwise absolute
                difference and pairwise elementwise product (à la arXiv:1705.02364).
                Default: `true`.
            subrepresentation (SubRepresentation): the sub-representation used for
                 the inputs. If `subrepresentation_right` is not given, then this
                 representation is used for both inputs with tied weights.
            subrepresentation_right (Optional[SubRepresentation]):
                 the sub-representation used for the right input. Optional.
                 If missing, `subrepresentation` is used with tied weights.
                 Default: `None`.
        """

        subrepresentation: SubRepresentation = BiLSTMDocAttention.Config()
        subrepresentation_right: Optional[SubRepresentation] = None
        encode_relations: bool = True

    def __init__(self, config: Config, embed_dim: Tuple[int, ...]) -> None:
        super().__init__(config)

        assert len(embed_dim) == 2

        if config.subrepresentation_right is not None:
            self.subrepresentations = nn.ModuleList(
                [
                    create_module(config.subrepresentation, embed_dim=embed_dim[0]),
                    create_module(
                        config.subrepresentation_right, embed_dim=embed_dim[1]
                    ),
                ]
            )
            if config.encode_relations:
                assert (
                    self.subrepresentations[0].representation_dim
                    == self.subrepresentations[1].representation_dim
                ), (
                    "Representations must have the same dimension"
                    ", because `encode_relations` involves elementwise operations."
                )
        else:
            assert embed_dim[0] == embed_dim[1], (
                "Embeddings must have the same dimension"
                ", because subrepresentation weights are tied."
            )
            subrep = create_module(config.subrepresentation, embed_dim=embed_dim[0])
            self.subrepresentations = nn.ModuleList([subrep, subrep])

        self.encode_relations = config.encode_relations
        self.representation_dim = self.subrepresentations[0].representation_dim
        if self.encode_relations:
            self.representation_dim *= 4
        else:
            self.representation_dim += self.subrepresentations[1].representation_dim

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
    ) -> torch.Tensor:
        """Computes the pair representations.

        Arguments:
            embeddings: token embeddings of the left sentence, followed by the
              token embeddings of the right sentence(s).
            lengths: the corresponding sequence lengths.

        Returns:
            A tensor of shape `(num_right_inputs, batch_size, rep_size)`, with
            the first dimension squeezed if one.
        """
        left_rep = self._represent(
            self.subrepresentations[0], embeddings[0], lengths[0]
        )
        assert len(embeddings) == len(lengths) and len(embeddings) >= 2

        # The leftmost inputs already came sorted by length. The others need to
        # be sorted as well, for packing. We do it manually.
        sorted_right_inputs = []
        sorted_right_indices = []
        for embs, lens in zip(embeddings[1:], lengths[1:]):
            lens_sorted, sorted_idx = lens.sort(descending=True)
            embs_sorted = embs[sorted_idx]
            sorted_right_inputs.append((embs_sorted, lens_sorted))
            sorted_right_indices.append(sorted_idx)
        sorted_right_reps = [
            self._represent(self.subrepresentations[1], embs, lens)
            for (embs, lens) in sorted_right_inputs
        ]
        # Put the right inputs back in the original order, so they still match
        # up within the batch to the left inputs
        right_reps = []
        for idx, rep in zip(sorted_right_indices, sorted_right_reps):
            _, desorted_idx = idx.sort()
            right_reps.append(rep[desorted_idx])

        final_reps = []
        for right_rep in right_reps:
            this_rep = []
            this_rep.append(left_rep)
            this_rep.append(right_rep)
            if self.encode_relations:
                this_rep.append(torch.abs(left_rep - right_rep))
                this_rep.append(left_rep * right_rep)
            final_reps.append(torch.cat(this_rep, -1))

        return torch.stack(final_reps).squeeze(0)
