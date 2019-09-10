#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union

from pytext.data.tensorizers import (
    ByteTokenTensorizer,
    SlotLabelTensorizer,
    TokenTensorizer,
)
from pytext.data.utils import UNK
from pytext.exporters.exporter import ModelExporter
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.embeddings import CharacterEmbedding, WordEmbedding
from pytext.models.model import Model
from pytext.models.module import create_module
from pytext.models.output_layers import CRFOutputLayer, WordTaggingOutputLayer
from pytext.models.representations.bilstm_slot_attn import BiLSTMSlotAttention
from pytext.models.representations.biseqcnn import BSeqCNNRepresentation
from pytext.models.representations.deepcnn import DeepCNNRepresentation
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

    __EXPANSIBLE__ = True

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
            DeepCNNRepresentation.Config,
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

    def get_export_input_names(self, tensorizers):
        return ["tokens_vals", "tokens_lens"]

    def get_export_output_names(self, tensorizers):
        return ["word_scores"]

    def vocab_to_export(self, tensorizers):
        return {"tokens_vals": list(tensorizers["tokens"].vocab)}

    def arrange_model_context(self, tensor_dict):
        return {"seq_lens": tensor_dict["tokens"][1]}

    def caffe2_export(self, tensorizers, tensor_dict, path, export_onnx_path=None):
        exporter = ModelExporter(
            ModelExporter.Config(),
            self.get_export_input_names(tensorizers),
            self.arrange_model_inputs(tensor_dict),
            self.vocab_to_export(tensorizers),
            self.get_export_output_names(tensorizers),
        )
        return exporter.export_to_caffe2(self, path, export_onnx_path=export_onnx_path)


class WordTaggingLiteModel(WordTaggingModel):
    """
    Also a word tagging model, but uses bytes as inputs to the model. Using
    bytes instead of words, the model does not need to store a word embedding
    table mapping words in the vocab to their embedding vector representations,
    but instead compute them on the fly using CharacterEmbedding. This produces
    an exported/serialized model that requires much less storage space as well
    as less memory during run/inference time.
    """

    class Config(WordTaggingModel.Config):
        class ByteModelInput(Model.Config.ModelInput):
            # We should support characters as well, but CharacterTokenTensorizer
            # does not support adding characters to vocab yet.
            token_bytes: ByteTokenTensorizer.Config = ByteTokenTensorizer.Config()
            labels: SlotLabelTensorizer.Config = SlotLabelTensorizer.Config()

        inputs: ByteModelInput = ByteModelInput()
        embedding: CharacterEmbedding.Config = CharacterEmbedding.Config()

    @classmethod
    def create_embedding(cls, config, tensorizers):
        return CharacterEmbedding(
            tensorizers["token_bytes"].NUM_BYTES,
            config.embedding.embed_dim,
            config.embedding.cnn.kernel_num,
            config.embedding.cnn.kernel_sizes,
            config.embedding.highway_layers,
            config.embedding.projection_dim,
        )

    def vocab_to_export(self, tensorizers):
        return {}

    def get_export_input_names(self, tensorizers):
        return ["token_bytes", "token_lens"]

    def arrange_model_inputs(self, tensor_dict):
        token_bytes, tokens_lens, _ = tensor_dict["token_bytes"]
        return (token_bytes, tokens_lens)

    def arrange_model_context(self, tensor_dict):
        return {"seq_lens": tensor_dict["token_bytes"][1]}
