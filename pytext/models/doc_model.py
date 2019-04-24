#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Union

import torch
from pytext.config import ConfigBase
from pytext.config.component import create_loss
from pytext.data.tensorizers import (
    LabelTensorizer,
    NumericLabelTensorizer,
    RawString,
    Tensorizer,
    TokenTensorizer,
)
from pytext.data.utils import PAD, UNK
from pytext.exporters.exporter import ModelExporter
from pytext.loss import BinaryCrossEntropyLoss
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.embeddings import WordEmbedding
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
from pytext.utils.torch import Vocabulary, list_max
from torch import jit


class DocModel(Model):
    """
    An n-ary document classification model. It can be used for all text
    classification scenarios. It supports :class:`~PureDocAttention`,
    :class:`~BiLSTMDocAttention` and :class:`~DocNNRepresentation` as the ways
    to represent the document followed by multi-layer perceptron (:class:`~MLPDecoder`)
    for projecting the document representation into label/target space.

    It can be instantiated just like any other :class:`~Model`.
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


class NewDocModel(DocModel):
    """DocModel that's compatible with the new Model abstraction, which is responsible
    for describing which inputs it expects and arranging its input tensors."""

    __EXPANSIBLE__ = True

    class Config(DocModel.Config):
        class ModelInput(Model.Config.ModelInput):
            tokens: TokenTensorizer.Config = TokenTensorizer.Config()
            labels: LabelTensorizer.Config = LabelTensorizer.Config(allow_unknown=True)
            # for metric reporter
            raw_text: RawString.Config = RawString.Config(column="text")

        inputs: ModelInput = ModelInput()
        embedding: WordEmbedding.Config = WordEmbedding.Config()

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens = tensor_dict["tokens"]
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
                word_ids = self.vocab.lookup_indices_2d(tokens)

                seq_lens = jit.annotate(List[int], [])

                for sentence in word_ids:
                    seq_lens.append(len(sentence))
                pad_to_length = list_max(seq_lens)
                for sentence in word_ids:
                    for _ in range(pad_to_length - len(sentence)):
                        sentence.append(self.pad_idx)

                logits = self.model(torch.tensor(word_ids), torch.tensor(seq_lens))
                return self.output_layer(logits)

        return Model()

    @classmethod
    def create_embedding(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        vocab = tensorizers["tokens"].vocab
        return WordEmbedding(
            len(vocab), config.embedding.embed_dim, None, None, vocab.idx[UNK], []
        )

    @classmethod
    def create_decoder(cls, config: Config, representation_dim: int, num_labels: int):
        return create_module(
            config.decoder, in_dim=representation_dim, out_dim=num_labels
        )

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        labels = tensorizers["labels"].labels
        embedding = cls.create_embedding(config, tensorizers)
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        decoder = cls.create_decoder(
            config, representation.representation_dim, len(labels)
        )
        # TODO change from_config function of ClassificationOutputLayer after migriting to new design
        loss = create_loss(config.output_layer.loss)
        output_layer_cls = (
            BinaryClassificationOutputLayer
            if isinstance(loss, BinaryCrossEntropyLoss)
            else MulticlassOutputLayer
        )
        output_layer = output_layer_cls(list(labels), loss)
        return cls(embedding, representation, decoder, output_layer)


class NewDocRegressionModel(NewDocModel):
    """
    Model that's compatible with the new Model abstraction, and is configured for
    regression tasks (specifically for labels, predictions, and loss).
    """

    class Config(NewDocModel.Config):
        class RegressionModelInput(Model.Config.ModelInput):
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
