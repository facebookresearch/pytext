#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union

from pytext.data.tensorizers import SlotLabelTensorizer, TokenTensorizer
from pytext.data.utils import UNK
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.embeddings import WordEmbedding
from pytext.models.model import Model
from pytext.models.module import create_module
from pytext.models.output_layers import CRFOutputLayer, WordTaggingOutputLayer
from pytext.models.representations.bilstm_slot_attn import BiLSTMSlotAttention
from pytext.models.representations.biseqcnn import BSeqCNNRepresentation
from pytext.models.representations.pass_through import PassThroughRepresentation


class WordTaggingModel_Deprecated(Model):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CRF module has parameters and it's forward function is not called in
        # model's forward function because of ONNX compatibility issue. This will
        # not work with DDP, thus setting find_unused_parameters to False to work
        # around, can be removed once DDP support params not used in model forward
        # function
        if isinstance(self.output_layer, CRFOutputLayer):
            self.find_unused_parameters = False


class WordTaggingModel(Model):
    class Config(Model.Config):
        class ModelInput(Model.Config.ModelInput):
            tokens: TokenTensorizer.Config = TokenTensorizer.Config()
            labels: SlotLabelTensorizer.Config = SlotLabelTensorizer.Config()

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
        output_layer = create_module(config.output_layer, labels=labels)
        return cls(embedding, representation, decoder, output_layer)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CRF module has parameters and it's forward function is not called in
        # model's forward function because of ONNX compatibility issue. This will
        # not work with DDP, thus setting find_unused_parameters to False to work
        # around, can be removed once DDP support params not used in model forward
        # function
        if isinstance(self.output_layer, CRFOutputLayer):
            self.find_unused_parameters = False

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens, _ = tensor_dict["tokens"]
        return (tokens, seq_lens)

    def arrange_targets(self, tensor_dict):
        return tensor_dict["labels"]
