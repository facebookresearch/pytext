#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Union

from pytext.config import ConfigBase
from pytext.config.field_config import WordFeatConfig
from pytext.data.tensorizers import (
    LabelTensorizer,
    MetaInput,
    Tensorizer,
    WordTensorizer,
)
from pytext.data.utils import UNK
from pytext.exporters.exporter import ModelExporter
from pytext.loss import CrossEntropyLoss
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.embeddings import WordEmbedding
from pytext.models.model import Model
from pytext.models.module import create_module
from pytext.models.output_layers import ClassificationOutputLayer
from pytext.models.representations.bilstm_doc_attention import BiLSTMDocAttention
from pytext.models.representations.docnn import DocNNRepresentation
from pytext.models.representations.pure_doc_attention import PureDocAttention


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

    class Config(DocModel.Config):
        class ModelInput(Model.Config.ModelInput):
            tokens: WordTensorizer.Config = WordTensorizer.Config()
            labels: LabelTensorizer.Config = LabelTensorizer.Config(allow_unknown=True)
            # for metric reporter
            raw_text: MetaInput.Config = MetaInput.Config(column="text")

        inputs: ModelInput = ModelInput()
        embedding: WordEmbedding.Config = WordEmbedding.Config()

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens = tensor_dict["tokens"]
        return (tokens, seq_lens)

    def arrange_targets(self, tensor_dict):
        return tensor_dict["labels"]

    def caffe2_export(self, tensorizers, tensor_dict, path, export_onnx_path=None):
        exporter = ModelExporter(
            ModelExporter.Config(),
            ["tokens", "tokens_lens"],
            self.arrange_model_inputs(tensor_dict),
            {"tokens": list(tensorizers["tokens"].vocab)},
            ["scores"],
        )
        return exporter.export_to_caffe2(self, path, export_onnx_path=export_onnx_path)

    @classmethod
    def create_embedding(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        vocab = tensorizers["tokens"].vocab
        return WordEmbedding(
            len(vocab), config.embedding.embed_dim, None, None, vocab.idx[UNK], []
        )

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        labels = tensorizers["labels"].labels
        embedding = cls.create_embedding(config, tensorizers)
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        decoder = create_module(
            config.decoder,
            in_dim=representation.representation_dim,
            out_dim=len(labels),
        )
        output_layer = ClassificationOutputLayer(list(labels), CrossEntropyLoss(None))
        return cls(embedding, representation, decoder, output_layer)
