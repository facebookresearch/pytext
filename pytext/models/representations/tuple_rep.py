#!/usr/bin/env python3

import itertools
from typing import Tuple, Union

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.models.module import create_module
from scipy.special import comb

from .bilstm_doc_attention import BiLSTMDocAttention
from .docnn import DocNNRepresentation
from .representation_base import RepresentationBase


# NOTE: the edge case of loading two distinct untied subrepresentations via the
# `load_path` config option is not currently supported.
class TupleRepresentation(RepresentationBase):
    """Wrapper representation for a tuple of inputs.

    Parameters:
        * `tied_weights`: whether to use the same representation, with tied weights,
          for all inputs. Default: `true`.
        * `encode_relations`: if `false`, return the concatenation of all the
          representations; if `true`, also concatenate their pairwise absolute
          differences and pairwise elementwise products (à la
          <https://arxiv.org/pdf/1705.02364.pdf>). Default: `true`.
        * `num_subrepresentations`: the arity. Default: 2.
    """

    class Config(RepresentationBase.Config, ConfigBase):
        subrepresentation: Union[
            BiLSTMDocAttention.Config, DocNNRepresentation.Config
        ] = BiLSTMDocAttention.Config()
        tied_weights: bool = True
        num_subrepresentations: int = 2
        encode_relations: bool = True

    def __init__(self, config: Config, embed_dim: Tuple[int, ...]) -> None:
        super().__init__(config)
        assert (
            config.num_subrepresentations > 1
            and config.num_subrepresentations == len(embed_dim)
        )

        if config.tied_weights:
            self.subrepresentations = nn.ModuleList(
                itertools.repeat(
                    create_module(config.subrepresentation, embed_dim=embed_dim[0]),
                    config.num_subrepresentations,
                )
            )
        else:
            self.subrepresentations = nn.ModuleList(
                create_module(config.subrepresentation, embed_dim=ed)
                for ed in embed_dim
            )

        self.representation_dim = self.subrepresentations[0].representation_dim
        self.encode_relations = config.encode_relations
        if self.encode_relations:
            self.representation_dim = (
                config.num_subrepresentations * self.representation_dim
                + 2
                * comb(config.num_subrepresentations, 2, exact=True)
                * self.representation_dim
            )
        else:
            self.representation_dim = (
                config.num_subrepresentations * self.representation_dim
            )

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
    ) -> Tuple[torch.Tensor, ...]:
        # The leftmost inputs already come sorted by length. The others need to
        # be sorted as well, for packing. We do it manually.
        sorted_inputs = [(embeddings[0], lengths[0])]
        sorted_indices = [None]
        for embs, lens in zip(embeddings[1:], lengths[1:]):
            lens_sorted, sorted_idx = lens.sort(descending=True)
            embs_sorted = embs[sorted_idx]
            sorted_inputs.append((embs_sorted, lens_sorted))
            sorted_indices.append(sorted_idx)
        representations = [
            self._represent(rep, embs, lens)
            for rep, (embs, lens) in zip(self.subrepresentations, sorted_inputs)
        ]

        # Put the inputs back in the original order, so they still match up to
        # each other as well as the targets.
        unsorted_representations = [representations[0]]
        for sorted_idx, rep in zip(sorted_indices[1:], representations[1:]):
            _, unsorted_idx = sorted_idx.sort()
            unsorted_representations.append(rep[unsorted_idx])

        if self.encode_relations:
            for rep_l, rep_r in itertools.combinations(unsorted_representations, 2):
                unsorted_representations.append(torch.abs(rep_l - rep_r))
                unsorted_representations.append(rep_l * rep_r)

        return torch.cat(unsorted_representations, -1)
