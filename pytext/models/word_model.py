#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union

from pytext.data.tensorizers import RawString, TokenTensorizer, WordLabelTensorizer
from pytext.data.utils import UNK
from pytext.loss import CrossEntropyLoss
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.embeddings import WordEmbedding
from pytext.models.model import Model
from pytext.models.module import create_module
from pytext.models.output_layers import CRFOutputLayer, WordTaggingOutputLayer
from pytext.models.representations.bilstm_slot_attn import BiLSTMSlotAttention
from pytext.models.representations.biseqcnn import BSeqCNNRepresentation
from pytext.models.representations.pass_through import PassThroughRepresentation


class WordTaggingModel(Model):
    """
    Word tagging model. It can be used for any task that requires predicting the
    tag for a word/token. For example, the following tasks can be modeled as word
    tagging tasks. This is not an exhaustive list.
    1. Part of speech tagging.
    2. Named entity recognition.
    3. Slot filling for task oriented dialog.

    It can be instantiated just like any other :class:`~Model`.
    """

    class Config(Model.Config):
        representation: Union[
            BiLSTMSlotAttention.Config,
            BSeqCNNRepresentation.Config,
            PassThroughRepresentation.Config,
        ] = BiLSTMSlotAttention.Config()
        output_layer: Union[
            WordTaggingOutputLayer.Config, CRFOutputLayer.Config
        ] = WordTaggingOutputLayer.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()


class NewWordTaggingModel(Model):
    class Config(Model.Config):
        class ModelInput(Model.Config.ModelInput):
            tokens: TokenTensorizer.Config = TokenTensorizer.Config()
            labels: WordLabelTensorizer.Config = WordLabelTensorizer.Config()
            # for metric reporter
            raw_text: RawString.Config = RawString.Config(column="text")

        inputs: ModelInput = ModelInput()
        embedding: WordEmbedding.Config = WordEmbedding.Config()

        representation: Union[
            BiLSTMSlotAttention.Config,  # TODO: make default when sorting solved
            BSeqCNNRepresentation.Config,
            PassThroughRepresentation.Config,
        ] = PassThroughRepresentation.Config()
        output_layer: Union[
            WordTaggingOutputLayer.Config, CRFOutputLayer.Config
        ] = WordTaggingOutputLayer.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()

    @classmethod
    def create_embedding(cls, config, tensorizers):
        vocab = tensorizers["tokens"].vocab
        return WordEmbedding(
            len(vocab), config.embedding.embed_dim, None, None, vocab.idx[UNK], []
        )

    @classmethod
    def from_config(cls, config, tensorizers):
        labels = tensorizers["labels"].vocab
        embedding = cls.create_embedding(config, tensorizers)
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        decoder = create_module(
            config.decoder,
            in_dim=representation.representation_dim,
            out_dim=len(labels),
        )
        # TODO after migration: create_module(config.output_layer, tensorizers=tensorizers)
        output_layer = WordTaggingOutputLayer(labels, CrossEntropyLoss(None))
        return cls(embedding, representation, decoder, output_layer)

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens = tensor_dict["tokens"]
        return (tokens, seq_lens)

    def arrange_targets(self, tensor_dict):
        return tensor_dict["labels"]
