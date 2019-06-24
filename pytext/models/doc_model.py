#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Union

import torch
from pytext.config import ConfigBase
from pytext.config.component import create_loss
from pytext.data.tensorizers import (
    ByteTokenTensorizer,
    LabelTensorizer,
    NumericLabelTensorizer,
    Tensorizer,
    TokenTensorizer,
)
from pytext.data.utils import PAD, UNK
from pytext.exporters.exporter import ModelExporter
from pytext.loss import BinaryCrossEntropyLoss
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.embeddings import CharacterEmbedding, EmbeddingList, WordEmbedding
from pytext.models.model import Model
from pytext.models.module import create_module
from pytext.models.output_layers import ClassificationOutputLayer, RegressionOutputLayer
from pytext.models.output_layers.doc_classification_output_layer import (
    BinaryClassificationOutputLayer,
    MulticlassOutputLayer,
)
from pytext.models.representations.bilstm_doc_attention import BiLSTMDocAttention
from pytext.models.representations.docnn import DocNNRepresentation
from pytext.models.representations.pure_doc_attention import PureDocAttention
from pytext.utils.torch import (
    Vocabulary,
    make_byte_inputs,
    make_sequence_lengths,
    pad_2d,
)
from torch import jit


class DocModel_Deprecated(Model):
    """
    An n-ary document classification model. It can be used for all text
    classification scenarios. It supports :class:`~PureDocAttention`,
    :class:`~BiLSTMDocAttention` and :class:`~DocNNRepresentation` as the ways
    to represent the document followed by multi-layer perceptron (:class:`~MLPDecoder`)
    for projecting the document representation into label/target space.

    It can be instantiated just like any other :class:`~Model`.

    DEPRECATED: Use DocModel instead
    """

    class Config(ConfigBase):
        representation: Union[
            PureDocAttention.Config,
            BiLSTMDocAttention.Config,
            DocNNRepresentation.Config,
        ] = BiLSTMDocAttention.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )


class DocModel(Model):
    """DocModel that's compatible with the new Model abstraction, which is responsible
    for describing which inputs it expects and arranging its input tensors."""

    __EXPANSIBLE__ = True

    class Config(Model.Config):
        class ModelInput(Model.Config.ModelInput):
            tokens: TokenTensorizer.Config = TokenTensorizer.Config()
            labels: LabelTensorizer.Config = LabelTensorizer.Config()

        inputs: ModelInput = ModelInput()
        embedding: WordEmbedding.Config = WordEmbedding.Config()
        representation: Union[
            PureDocAttention.Config,
            BiLSTMDocAttention.Config,
            DocNNRepresentation.Config,
        ] = BiLSTMDocAttention.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens, _ = tensor_dict["tokens"]
        return (tokens, seq_lens)

    def arrange_targets(self, tensor_dict):
        return tensor_dict["labels"]

    def get_export_input_names(self, tensorizers):
        return ["tokens", "tokens_lens"]

    def get_export_output_names(self, tensorizers):
        return ["scores"]

    def vocab_to_export(self, tensorizers):
        return {"tokens": list(tensorizers["tokens"].vocab)}

    def caffe2_export(self, tensorizers, tensor_dict, path, export_onnx_path=None):
        exporter = ModelExporter(
            ModelExporter.Config(),
            self.get_export_input_names(tensorizers),
            self.arrange_model_inputs(tensor_dict),
            self.vocab_to_export(tensorizers),
            self.get_export_output_names(tensorizers),
        )
        return exporter.export_to_caffe2(self, path, export_onnx_path=export_onnx_path)

    def torchscriptify(self, tensorizers, traced_model):
        output_layer = self.output_layer.torchscript_predictions()

        input_vocab = tensorizers["tokens"].vocab

        class Model(jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.vocab = Vocabulary(input_vocab, unk_idx=input_vocab.idx[UNK])
                self.model = traced_model
                self.output_layer = output_layer
                self.pad_idx = jit.Attribute(input_vocab.idx[PAD], int)

            @jit.script_method
            def forward(self, tokens: List[List[str]]):
                seq_lens = make_sequence_lengths(tokens)
                word_ids = self.vocab.lookup_indices_2d(tokens)
                word_ids = pad_2d(word_ids, seq_lens, self.pad_idx)
                logits = self.model(torch.tensor(word_ids), torch.tensor(seq_lens))
                return self.output_layer(logits)

        return Model()

    @classmethod
    def create_embedding(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        return create_module(
            config.embedding,
            tensorizer=tensorizers["tokens"],
            init_from_saved_state=config.init_from_saved_state,
        )

    @classmethod
    def create_decoder(cls, config: Config, representation_dim: int, num_labels: int):
        return create_module(
            config.decoder, in_dim=representation_dim, out_dim=num_labels
        )

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        labels = tensorizers["labels"].vocab
        embedding = cls.create_embedding(config, tensorizers)
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        decoder = cls.create_decoder(
            config, representation.representation_dim, len(labels)
        )
        loss = create_loss(config.output_layer.loss)
        output_layer_cls = (
            BinaryClassificationOutputLayer
            if isinstance(loss, BinaryCrossEntropyLoss)
            else MulticlassOutputLayer
        )
        output_layer = output_layer_cls(list(labels), loss)
        return cls(embedding, representation, decoder, output_layer)


class ByteTokensDocumentModel(DocModel):
    """
    DocModel that receives both word IDs and byte IDs as inputs (concatenating
    word and byte-token embeddings to represent input tokens).
    """

    class Config(DocModel.Config):
        class ByteModelInput(DocModel.Config.ModelInput):
            token_bytes: ByteTokenTensorizer.Config = ByteTokenTensorizer.Config()

        inputs: ByteModelInput = ByteModelInput()
        byte_embedding: CharacterEmbedding.Config = CharacterEmbedding.Config()

    @classmethod
    def create_embedding(cls, config, tensorizers: Dict[str, Tensorizer]):
        word_tensorizer = config.inputs.tokens
        byte_tensorizer = config.inputs.token_bytes
        assert word_tensorizer.column == byte_tensorizer.column
        assert word_tensorizer.tokenizer.items() == byte_tensorizer.tokenizer.items()
        assert word_tensorizer.max_seq_len == byte_tensorizer.max_seq_len

        word_embedding = create_module(
            config.embedding, tensorizer=tensorizers["tokens"]
        )
        byte_embedding = CharacterEmbedding(
            ByteTokenTensorizer.NUM_BYTES,
            config.byte_embedding.embed_dim,
            config.byte_embedding.cnn.kernel_num,
            config.byte_embedding.cnn.kernel_sizes,
            config.byte_embedding.highway_layers,
            config.byte_embedding.projection_dim,
        )
        return EmbeddingList([word_embedding, byte_embedding], concat=True)

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens, _ = tensor_dict["tokens"]
        token_bytes, byte_seq_lens, _ = tensor_dict["token_bytes"]
        assert (seq_lens == byte_seq_lens).all().item()
        return tokens, token_bytes, seq_lens

    def get_export_input_names(self, tensorizers):
        return ["tokens", "token_bytes", "tokens_lens"]

    def torchscriptify(self, tensorizers, traced_model):
        output_layer = self.output_layer.torchscript_predictions()
        max_byte_len = tensorizers["token_bytes"].max_byte_len
        byte_offset_for_non_padding = tensorizers["token_bytes"].offset_for_non_padding
        input_vocab = tensorizers["tokens"].vocab

        class Model(jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.vocab = Vocabulary(input_vocab, unk_idx=input_vocab.idx[UNK])
                self.max_byte_len = jit.Attribute(max_byte_len, int)
                self.byte_offset_for_non_padding = jit.Attribute(
                    byte_offset_for_non_padding, int
                )
                self.pad_idx = jit.Attribute(input_vocab.idx[PAD], int)
                self.model = traced_model
                self.output_layer = output_layer

            @jit.script_method
            def forward(self, tokens: List[List[str]]):
                seq_lens = make_sequence_lengths(tokens)
                word_ids = self.vocab.lookup_indices_2d(tokens)
                word_ids = pad_2d(word_ids, seq_lens, self.pad_idx)
                token_bytes, _ = make_byte_inputs(
                    tokens, self.max_byte_len, self.byte_offset_for_non_padding
                )
                logits = self.model(
                    torch.tensor(word_ids), token_bytes, torch.tensor(seq_lens)
                )
                return self.output_layer(logits)

        return Model()


class DocRegressionModel(DocModel):
    """
    Model that's compatible with the new Model abstraction, and is configured for
    regression tasks (specifically for labels, predictions, and loss).
    """

    class Config(DocModel.Config):
        class RegressionModelInput(DocModel.Config.ModelInput):
            tokens: TokenTensorizer.Config = TokenTensorizer.Config()
            labels: NumericLabelTensorizer.Config = NumericLabelTensorizer.Config()

        inputs: RegressionModelInput = RegressionModelInput()
        output_layer: RegressionOutputLayer.Config = RegressionOutputLayer.Config()

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        embedding = cls.create_embedding(config, tensorizers)
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        decoder = create_module(
            config.decoder, in_dim=representation.representation_dim, out_dim=1
        )
        output_layer = RegressionOutputLayer.from_config(config.output_layer)
        return cls(embedding, representation, decoder, output_layer)
