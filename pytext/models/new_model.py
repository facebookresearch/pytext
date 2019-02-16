#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict

from pytext.config.component import Component, ComponentType
from pytext.config.field_config import WordFeatConfig
from pytext.data.tensorizers import LabelTensorizer, Tensorizer, WordTensorizer
from pytext.data.utils import UNK
from pytext.loss import CrossEntropyLoss
from pytext.models import doc_model
from pytext.models.doc_model import ClassificationOutputLayer
from pytext.models.embeddings import WordEmbedding
from pytext.models.module import create_module


class Model(Component):
    __COMPONENT_TYPE__ = ComponentType.MODEL2
    __EXPANSIBLE__ = True

    class Config(Component.Config):
        inputs: Dict[str, Tensorizer.Config]

    def build_context_for_metrics(self, batch):
        return {}

    def train_batch(self, batch):
        model_inputs = self.arrange_model_inputs(batch)
        model_outputs = self(*model_inputs)
        loss = self.get_loss(model_outputs, self.arrange_targets(batch), None)
        predictions, scores = self.get_pred(model_outputs)
        targets = self.arrange_targets(batch)
        # These are another reason I think it might make sense for model
        # to own metric reporting
        metric_data = (predictions, targets, scores, loss, model_inputs)
        return loss, metric_data

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        raise NotImplementedError


class DocModel(Model, doc_model.DocModel):
    class Config(Model.Config, doc_model.DocModel.Config):
        inputs: Dict[str, Tensorizer.Config] = {
            "tokens": WordTensorizer.Config(),
            "labels": LabelTensorizer.Config(),
        }
        embedding: WordFeatConfig = WordFeatConfig()

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens = tensor_dict["tokens"]
        return (tokens, seq_lens)

    def arrange_targets(self, tensor_dict):
        return tensor_dict["labels"]

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        vocab = tensorizers["tokens"].vocab
        labels = tensorizers["labels"].labels

        embedding = WordEmbedding(
            len(vocab), config.embedding.embed_dim, None, None, vocab.idx[UNK], []
        )
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        decoder = create_module(
            config.decoder,
            in_dim=representation.representation_dim,
            out_dim=len(labels),
        )
        output_layer = ClassificationOutputLayer(labels, CrossEntropyLoss(None))
        return cls(embedding, representation, decoder, output_layer)
