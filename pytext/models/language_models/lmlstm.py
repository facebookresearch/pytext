#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional, Tuple, Union

import torch
from pytext.config import ConfigBase
from pytext.config.module_config import ExporterType
from pytext.data import CommonMetadata
from pytext.data.tensorizers import Tensorizer, TokenTensorizer
from pytext.exporters.custom_exporters import get_exporter
from pytext.exporters.exporter import ModelExporter
from pytext.models.decoders import DecoderBase
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.embeddings import EmbeddingBase
from pytext.models.embeddings.word_embedding import WordEmbedding
from pytext.models.model import BaseModel, Model
from pytext.models.module import create_module
from pytext.models.output_layers import OutputLayerBase
from pytext.models.output_layers.lm_output_layer import LMOutputLayer
from pytext.models.representations.bilstm import BiLSTM
from pytext.models.representations.deepcnn import DeepCNNRepresentation as CNN
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


class LMLSTM(BaseModel):
    """
    `LMLSTM` implements a word-level language model that uses LSTMs to
    represent the document.

    """

    class Config(BaseModel.Config):
        class ModelInput(Model.Config.ModelInput):
            tokens: Optional[TokenTensorizer.Config] = TokenTensorizer.Config(
                add_bos_token=True, add_eos_token=True
            )

        inputs: ModelInput = ModelInput()
        embedding: WordEmbedding.Config = WordEmbedding.Config()
        representation: Union[BiLSTM.Config, CNN.Config] = BiLSTM.Config(
            bidirectional=False
        )
        decoder: Optional[MLPDecoder.Config] = MLPDecoder.Config()
        output_layer: LMOutputLayer.Config = LMOutputLayer.Config()
        tied_weights: bool = False
        stateful: bool = False
        caffe2_format: ExporterType = ExporterType.PREDICTOR

    @classmethod
    def checkTokenConfig(cls, tokens: Optional[TokenTensorizer.Config]):
        if tokens is None:
            raise ValueError(
                "Tokens cannot be None. Please set it to TokenTensorizer in"
                "config file."
            )

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        cls.checkTokenConfig(tensorizers["tokens"])
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
            if decoder.get_decoder()[0][-1].weight.size() != embedding.weight.size():
                raise ValueError(
                    "Embedding dimension must be same as representation "
                    "dimensions when using tied weights"
                )
            decoder.get_decoder()[0][-1].weight = embedding.weight
        output_layer = create_module(config.output_layer, labels=labels)
        exporter = get_exporter(config.caffe2_format)
        return cls(
            embedding=embedding,
            representation=representation,
            decoder=decoder,
            output_layer=output_layer,
            stateful=config.stateful,
            exporter=exporter,
        )

    def __init__(
        self,
        embedding: EmbeddingBase = Config.embedding,
        representation: RepresentationBase = Config.representation,
        decoder: DecoderBase = Config.decoder,
        output_layer: OutputLayerBase = Config.output_layer,
        stateful: bool = Config.stateful,
        exporter: object = ModelExporter,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.representation = representation
        self.decoder = decoder
        self.output_layer = output_layer
        self.stateful = stateful
        self.module_list = [embedding, representation, decoder]
        self._states: Optional[Tuple] = None
        self.exporter = exporter

    def cpu(self):
        if self.stateful and self._states:
            self._states = (self._states[0].cpu(), self._states[1].cpu())
        return self._apply(lambda t: t.cpu())

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens, _ = tensor_dict["tokens"]
        # Omit last token because it won't have a corresponding target
        return (tokens[:, 0:-1].contiguous(), seq_lens - 1)

    def arrange_targets(self, tensor_dict):
        # Omit first token because it won't have a corresponding input
        tokens, seq_lens, _ = tensor_dict["tokens"]
        return (tokens[:, 1:].contiguous(), seq_lens - 1)

    def get_export_input_names(self, tensorizers):
        return ["tokens_vals", "tokens_lens"]

    def get_export_output_names(self, tensorizers):
        return ["scores"]

    def vocab_to_export(self, tensorizers):
        return {"tokens_vals": list(tensorizers["tokens"].vocab)}

    def caffe2_export(self, tensorizers, tensor_dict, path, export_onnx_path=None):
        exporter = self.exporter(
            self.exporter.Config(),
            self.get_export_input_names(tensorizers),
            self.arrange_model_inputs(tensor_dict),
            self.vocab_to_export(tensorizers),
            self.get_export_output_names(tensorizers),
        )
        return exporter.export_to_caffe2(self, path, export_onnx_path=export_onnx_path)

    def forward(
        self, tokens: torch.Tensor, seq_len: torch.Tensor
    ) -> List[torch.Tensor]:
        token_emb = self.embedding(tokens)

        rep = None
        if isinstance(self.representation, BiLSTM):
            if self.stateful and self._states is None:
                self._states = self.init_hidden(tokens.size(0))
            rep, states = self.representation(token_emb, seq_len, states=self._states)
            if self.stateful:
                self._states = repackage_hidden(states)
        elif isinstance(self.representation, CNN):
            rep = self.representation(token_emb)

        if self.decoder is None:
            output = rep
        else:
            if not isinstance(rep, (list, tuple)):
                rep = [rep]
            output = self.decoder(*rep)

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
