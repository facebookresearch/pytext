#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Union

from pytext.config import ConfigBase
from pytext.config.contextual_intent_slot import ModelInput
from pytext.data import CommonMetadata
from pytext.data.tensorizers import (
    FloatTensorizer,
    LabelTensorizer,
    SlotLabelTensorizer,
    TokenTensorizer,
)
from pytext.data.utils import UNK
from pytext.models.embeddings import WordEmbedding
from pytext.models.model import Model
from pytext.models.module import create_module
from pytext.models.representations.pass_through import PassThroughRepresentation

from .decoders import IntentSlotModelDecoder
from .embeddings import EmbeddingList
from .output_layers.intent_slot_output_layer import IntentSlotOutputLayer
from .output_layers.word_tagging_output_layer import CRFOutputLayer
from .representations.bilstm_doc_slot_attention import BiLSTMDocSlotAttention
from .representations.jointcnn_rep import JointCNNRepresentation


class JointModel(Model):
    """
    A joint intent-slot model. This is framed as a model to do document
    classification model and word tagging tasks where the embedding and text
    representation layers are shared for both tasks.

    The supported representation layers are based on bidirectional LSTM or CNN.

    It can be instantiated just like any other :class:`~Model`.
    """

    class Config(ConfigBase):
        representation: Union[
            BiLSTMDocSlotAttention.Config, JointCNNRepresentation.Config
        ] = BiLSTMDocSlotAttention.Config()
        output_layer: IntentSlotOutputLayer.Config = (IntentSlotOutputLayer.Config())
        decoder: IntentSlotModelDecoder.Config = IntentSlotModelDecoder.Config()
        default_doc_loss_weight: float = 0.2
        default_word_loss_weight: float = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CRF module has parameters and it's forward function is not called in
        # model's forward function because of ONNX compatibility issue. This will
        # not work with DDP, thus setting find_unused_parameters to False to work
        # around, can be removed once DDP support params not used in model forward
        # function
        if isinstance(self.output_layer.word_output, CRFOutputLayer):
            self.find_unused_parameters = False

    @classmethod
    def from_config(cls, model_config, feat_config, metadata: CommonMetadata):
        embedding = cls.create_embedding(feat_config, metadata)
        representation = create_module(
            model_config.representation, embed_dim=embedding.embedding_dim
        )
        dense_feat_dim = 0
        for decoder_feat in (ModelInput.DENSE,):  # Only 1 right now.
            if getattr(feat_config, decoder_feat, False):
                dense_feat_dim = getattr(feat_config, ModelInput.DENSE).dim

        doc_label_meta, word_label_meta = metadata.target
        decoder = create_module(
            model_config.decoder,
            in_dim_doc=representation.doc_representation_dim + dense_feat_dim,
            in_dim_word=representation.word_representation_dim + dense_feat_dim,
            out_dim_doc=doc_label_meta.vocab_size,
            out_dim_word=word_label_meta.vocab_size,
        )

        if dense_feat_dim > 0:
            decoder.num_decoder_modules = 1
        output_layer = create_module(
            model_config.output_layer, doc_label_meta, word_label_meta
        )
        return cls(embedding, representation, decoder, output_layer)


class IntentSlotModel(Model):
    """
    A joint intent-slot model. This is framed as a model to do document
    classification model and word tagging tasks where the embedding and text
    representation layers are shared for both tasks.

    The supported representation layers are based on bidirectional LSTM or CNN.

    It can be instantiated just like any other :class:`~Model`.

    This is in the new data handling design involving tensorizers; that is the
    difference between this and JointModel
    """

    __EXPANSIBLE__ = True

    class Config(Model.Config):
        class ModelInput(Model.Config.ModelInput):
            tokens: TokenTensorizer.Config = TokenTensorizer.Config()
            word_labels: SlotLabelTensorizer.Config = SlotLabelTensorizer.Config(
                allow_unknown=True
            )
            doc_labels: LabelTensorizer.Config = LabelTensorizer.Config(
                allow_unknown=True
            )
            doc_weight: FloatTensorizer.Config = FloatTensorizer.Config(
                column="doc_weight"
            )
            word_weight: FloatTensorizer.Config = FloatTensorizer.Config(
                column="word_weight"
            )

        inputs: ModelInput = ModelInput()
        word_embedding: WordEmbedding.Config = WordEmbedding.Config()

        representation: Union[
            BiLSTMDocSlotAttention.Config,
            JointCNNRepresentation.Config,
            PassThroughRepresentation.Config,
        ] = BiLSTMDocSlotAttention.Config()
        output_layer: IntentSlotOutputLayer.Config = (IntentSlotOutputLayer.Config())
        decoder: IntentSlotModelDecoder.Config = IntentSlotModelDecoder.Config()
        default_doc_loss_weight: float = 0.2
        default_word_loss_weight: float = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CRF module has parameters and it's forward function is not called in
        # model's forward function because of ONNX compatibility issue. This will
        # not work with DDP, thus setting find_unused_parameters to False to work
        # around, can be removed once DDP support params not used in model forward
        # function
        self.find_unused_parameters = False

    @classmethod
    def create_embedding(cls, config, tensorizers):
        vocab = tensorizers["tokens"].vocab
        word_embedding = WordEmbedding(
            len(vocab), config.word_embedding.embed_dim, None, None, vocab.idx[UNK], []
        )
        return EmbeddingList([word_embedding], concat=True)

    @classmethod
    def from_config(cls, config, tensorizers):
        word_labels = tensorizers["word_labels"].vocab
        doc_labels = tensorizers["doc_labels"].vocab

        embedding = cls.create_embedding(config, tensorizers)
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )

        decoder = create_module(
            config.decoder,
            in_dim_doc=representation.doc_representation_dim,
            in_dim_word=representation.word_representation_dim,
            out_dim_doc=len(doc_labels),
            out_dim_word=len(word_labels),
        )

        output_layer = create_module(
            config.output_layer, doc_labels=doc_labels, word_labels=word_labels
        )

        return cls(embedding, representation, decoder, output_layer)

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens, _ = tensor_dict["tokens"]
        return (tokens, seq_lens)

    def arrange_targets(self, tensor_dict):
        intent_tensor = tensor_dict["doc_labels"]
        slot_tensor = tensor_dict["word_labels"]
        return intent_tensor, slot_tensor

    def vocab_to_export(self, tensorizers):
        return {"tokens": list(tensorizers["tokens"].vocab)}

    def get_export_input_names(self, tensorizers):
        return ["tokens", "tokens_lens"]

    def get_export_output_names(self, tensorizers):
        return ["doc_scores", "word_scores"]

    def arrange_model_context(self, tensor_dict):
        return {
            "doc_weight": tensor_dict["doc_weight"],
            "word_weight": tensor_dict["word_weight"],
            "seq_lens": tensor_dict["tokens"][1],
        }
