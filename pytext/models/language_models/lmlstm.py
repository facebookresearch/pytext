#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple, Union

import torch
from pytext.config import ConfigBase
from pytext.data import CommonMetadata
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.model import Model
from pytext.models.output_layers.lm_output_layer import LMOutputLayer
from pytext.models.representations.bilstm import BiLSTM


def repackage_hidden(
    hidden: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Wraps hidden states in new Tensors, to detach them from their history.

    Args:
        hidden (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): Tensor or a
            tuple of tensors to repackage.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, ...]]: Repackaged output

    """
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(v) for v in hidden)


class LMLSTM(Model):
    """
    `LMLSTM` implements a word-level language model that uses LSTMs to
    represent the document.

    """

    class Config(ConfigBase):
        """
        Configuration class for `LMLSTM`.

        Attributes:
            representation (BiLSTM.Config): Config for the BiLSTM representation.
            decoder (MLPDecoder.Config): Config for the MLP Decoder.
            output_layer (LMOutputLayer.Config): Config for the language model
                output layer.
            tied_weights (bool): If `True` use a common weights matrix between
                the word embeddings and the decoder. Defaults to `False`.
            stateful (bool): If `True`, do not reset hidden state of LSTM
                across batches.
        """

        representation: BiLSTM.Config = BiLSTM.Config(bidirectional=False)
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: LMOutputLayer.Config = LMOutputLayer.Config()
        tied_weights: bool = False
        stateful: bool = False

    @classmethod
    def from_config(cls, model_config, feat_config, metadata: CommonMetadata):
        """
        Factory method to construct an instance of LMLSTM from the module's
        config object and the field's metadata object.

        Args:
            config (LMLSTM.Config): Configuration object specifying all the
                parameters of LMLSTM.
            feat_config (FeatureConfig): Configuration object specifying all the
                parameters of all input features.
            metadata (FieldMeta): Object containing this field's metadata.

        Returns:
            type: An instance of `LMLSTM`.
        """
        model = super().from_config(model_config, feat_config, metadata)
        if model_config.tied_weights:
            if not feat_config.word_feat:
                raise ValueError(
                    "Word embeddings must be used when enabling tied weights"
                )
            elif (
                feat_config.word_feat.embed_dim
                != model.representation.representation_dim
            ):
                print(feat_config.word_feat.embed_dim)
                print(model.representation.representation_dim)
                raise ValueError(
                    "Embedding dimension must be same as representation "
                    "dimesnions when using tied weights"
                )
            model.decoder.get_decoder()[0].weight = model.embedding[0].weight

        # Setting an attribute on model object outside the class.
        model.tied_weights = model_config.tied_weights
        model.stateful = model_config.stateful
        return model

    def __init__(self, *inputs) -> None:
        super().__init__(*inputs)
        self._states: Optional[Tuple] = None

    def forward(self, tokens, *inputs) -> List[torch.Tensor]:
        # tokens dim: (bsz, max_seq_len) -> token_emb dim: (bsz, max_seq_len, dim)
        token_emb = self.embedding(tokens)
        if self.stateful and self._states is None:
            self._states = self.init_hidden(tokens.size(0))

        rep, states = self.representation(token_emb, *inputs, states=self._states)
        if self.decoder is None:
            output = rep
        else:
            if not isinstance(rep, (list, tuple)):
                rep = [rep]
            output = self.decoder(*rep)

        if self.stateful:
            self._states = repackage_hidden(states)
        return output  # (bsz, nclasses)

    def init_hidden(self, bsz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize the hidden states of the LSTM if the language model is
        stateful.

        Args:
            bsz (int): Batch size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initialized hidden state and
            cell state of the LSTM.
        """
        # TODO make hidden states initialization more generalized
        weight = next(self.parameters())
        num_layers = self.representation.num_layers
        rnn_hidden_dim = self.representation.representation_dim
        return (
            weight.new_zeros(bsz, num_layers, rnn_hidden_dim),
            weight.new_zeros(bsz, num_layers, rnn_hidden_dim),
        )
