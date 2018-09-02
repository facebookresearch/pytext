#!/usr/bin/env python3
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.data import CommonMetadata
from pytext.models.model import Model
from pytext.models.output_layer.lm_output_layer import LMOutputLayer
from pytext.models.projections.mlp_projection import MLPProjection
from pytext.models.representations.bilstm_self_attn import BiLSTMSelfAttention
from pytext.utils import cuda_utils


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class LMLSTM(Model):
    """A word-level language model that uses LSTM to represent the document."""

    class Config(ConfigBase):
        representation: BiLSTMSelfAttention.Config = BiLSTMSelfAttention.Config(
            self_attn_dim=0, bidirectional=False
        )
        output_config: LMOutputLayer.Config = LMOutputLayer.Config()
        proj_config: MLPProjection.Config = MLPProjection.Config()
        tied_weights: bool = False
        stateful: bool = False

    @classmethod
    def from_config(cls, model_config, feat_config, metadata: CommonMetadata):
        model = super().from_config(model_config, feat_config, metadata)
        if model_config.tied_weights:
            if not feat_config.word_feat:
                raise ValueError(
                    "Word embeddings must be used when enabling tied weights"
                )
            elif (
                model.embedding.word_embed.embedding_dim
                != model.representation.representation_dim
            ):
                print(model.embedding.word_embed.embedding_dim)
                print(model.representation.representation_dim)
                raise ValueError(
                    "Embedding dimension must be same as representation "
                    "dimesnions when using tied weights"
                )
            model.projection.get_projection()[
                0
            ].weight = model.embedding.word_embed.weight

        # Setting an attribute on model object outside the class.
        model.tied_weights = model_config.tied_weights
        model.stateful = model_config.stateful
        return model

    def __init__(self, *inputs) -> None:
        super().__init__(*inputs)
        self._states: Optional[Tuple] = None

    def get_model_params_for_optimizer(
        self
    ) -> Tuple[Dict[str, nn.Parameter], Dict[str, nn.Parameter]]:
        if self.tied_weights:
            return {}, {}  # Don't use SparseAdam when tying weights.
        return super().get_model_params_for_optimizer()

    def forward(
        self,
        tokens: torch.Tensor,
        tokens_lens: torch.Tensor,
        dict_feat: Tuple[torch.Tensor, ...] = None,
        cap_feat: Tuple[torch.Tensor, ...] = None,
        chars: torch.Tensor = None,
    ) -> List[torch.Tensor]:
        # tokens dim: (bsz, max_seq_len) -> token_emb dim: (bsz, max_seq_len, dim)
        token_emb = self.embedding(tokens, dict_feat, cap_feat, chars)
        if self.stateful and self._states is None:
            self._states = self.init_hidden(tokens.size(0))

        output, states = cuda_utils.parallelize(
            StatefulDataParallelModel(self.projection, self.representation),
            (token_emb, tokens_lens, self._states),
        )
        if self.stateful:
            self._states = repackage_hidden(states)
        return output  # (bsz, nclasses)

    def init_hidden(self, bsz):
        # TODO make hidden states initialization more generalized
        weight = next(self.parameters())
        num_layers = self.representation.lstm.num_layers
        rnn_hidden_dim = self.representation.representation_dim
        return (
            weight.new_zeros(bsz, num_layers, rnn_hidden_dim),
            weight.new_zeros(bsz, num_layers, rnn_hidden_dim),
        )


class StatefulDataParallelModel(nn.Module):
    def __init__(self, projection, representation):
        super().__init__()
        self.projection = projection
        self.representation = representation

    def forward(
        self,
        token_emb: torch.Tensor,
        tokens_lens: torch.Tensor,
        prev_state: torch.Tensor = None,
    ):
        rep, state = self.representation(token_emb, tokens_lens, prev_state)
        if not isinstance(rep, (list, tuple)):
            rep = [rep]

        return self.projection(*rep), state
