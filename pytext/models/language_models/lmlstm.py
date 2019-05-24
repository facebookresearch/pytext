#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional, Tuple, Union

import torch
from pytext.config import ConfigBase
from pytext.data import CommonMetadata
from pytext.data.tensorizers import Tensorizer, TokenTensorizer
from pytext.models.decoders import DecoderBase
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.embeddings import EmbeddingBase
from pytext.models.embeddings.word_embedding import WordEmbedding
from pytext.models.model import BaseModel, Model
from pytext.models.module import create_module
from pytext.models.output_layers import OutputLayerBase
from pytext.models.output_layers.lm_output_layer import LMOutputLayer
from pytext.models.representations.bilstm import BiLSTM
from pytext.models.representations.representation_base import RepresentationBase


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


class LMLSTM_Deprecated(Model):
    """
    `LMLSTM` implements a word-level language model that uses LSTMs to
    represent the document.

    DEPRECATED: Use LMLSTM instead
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
                    "dimensions when using tied weights"
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


class LMLSTM(BaseModel):
    """
    `LMLSTM` implements a word-level language model that uses LSTMs to
    represent the document.

    """

    class Config(BaseModel.Config):
        class ModelInput(Model.Config.ModelInput):
            tokens: TokenTensorizer.Config = TokenTensorizer.Config(
                add_bos_token=True, add_eos_token=True
            )

        inputs: ModelInput = ModelInput()
        embedding: WordEmbedding.Config = WordEmbedding.Config()
        representation: BiLSTM.Config = BiLSTM.Config(bidirectional=False)
        decoder: Optional[MLPDecoder.Config] = MLPDecoder.Config()
        output_layer: LMOutputLayer.Config = LMOutputLayer.Config()
        tied_weights: bool = False
        stateful: bool = False

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        embedding = create_module(config.embedding, tensorizer=tensorizers["tokens"])
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        labels = tensorizers["tokens"].vocab
        decoder = create_module(
            config.decoder,
            in_dim=representation.representation_dim,
            out_dim=len(labels),
        )
        if config.tied_weights:
            decoder.get_decoder()[0].weight = embedding.weight
        output_layer = create_module(config.output_layer, labels=labels)
        return cls(
            embedding=embedding,
            representation=representation,
            decoder=decoder,
            output_layer=output_layer,
            stateful=config.stateful,
        )

    def __init__(
        self,
        embedding: EmbeddingBase = Config.embedding,
        representation: RepresentationBase = Config.representation,
        decoder: DecoderBase = Config.decoder,
        output_layer: OutputLayerBase = Config.output_layer,
        stateful: bool = Config.stateful,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.representation = representation
        self.decoder = decoder
        self.output_layer = output_layer
        self.stateful = stateful
        self.module_list = [embedding, representation, decoder]
        self._states: Optional[Tuple] = None

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens = tensor_dict["tokens"]
        # Omit last token because it won't have a corresponding target
        return (tokens[:, 0:-1].contiguous(), seq_lens - 1)

    def arrange_targets(self, tensor_dict):
        # Omit first token because it won't have a corresponding input
        return tensor_dict["tokens"][0][:, 1:].contiguous()

    def get_export_input_names(self, tensorizers):
        return ["tokens", "tokens_lens"]

    def get_export_output_names(self, tensorizers):
        return ["scores"]

    def vocab_to_export(self, tensorizers):
        return {"tokens": list(tensorizers["tokens"].vocab)}

    def forward(
        self, tokens: torch.Tensor, seq_len: torch.Tensor
    ) -> List[torch.Tensor]:
        token_emb = self.embedding(tokens)
        if self.stateful and self._states is None:
            self._states = self.init_hidden(tokens.size(0))

        rep, states = self.representation(token_emb, seq_len, states=self._states)
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
        weight = next(self.parameters())
        num_layers = self.representation.lstm.num_layers
        rnn_hidden_dim = self.representation.representation_dim
        return (
            weight.new_zeros(bsz, num_layers, rnn_hidden_dim),
            weight.new_zeros(bsz, num_layers, rnn_hidden_dim),
        )
