#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional, Union

import pytext.utils.cuda as cuda_util
import torch
from pytext.common.constants import SpecialTokens
from pytext.data.tensorizers import (
    FloatTensorizer,
    LabelTensorizer,
    SlotLabelTensorizer,
    TokenTensorizer,
)
from pytext.exporters.exporter import ModelExporter
from pytext.models.embeddings import WordEmbedding
from pytext.models.model import Model
from pytext.models.module import create_module
from pytext.models.representations.pass_through import PassThroughRepresentation
from pytext.utils.usage import log_class_usage

from .decoders import IntentSlotModelDecoder
from .embeddings import EmbeddingList
from .output_layers.intent_slot_output_layer import IntentSlotOutputLayer
from .representations.bilstm_doc_slot_attention import BiLSTMDocSlotAttention
from .representations.jointcnn_rep import (
    JointCNNRepresentation,
    SharedCNNRepresentation,
)


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
            doc_weight: Optional[FloatTensorizer.Config] = None
            word_weight: Optional[FloatTensorizer.Config] = None

        inputs: ModelInput = ModelInput()
        word_embedding: WordEmbedding.Config = WordEmbedding.Config()

        representation: Union[
            BiLSTMDocSlotAttention.Config,
            JointCNNRepresentation.Config,
            SharedCNNRepresentation.Config,
            PassThroughRepresentation.Config,
        ] = BiLSTMDocSlotAttention.Config()
        output_layer: IntentSlotOutputLayer.Config = (IntentSlotOutputLayer.Config())
        decoder: IntentSlotModelDecoder.Config = IntentSlotModelDecoder.Config()
        default_doc_loss_weight: float = 0.2
        default_word_loss_weight: float = 0.5

    def __init__(
        self, default_doc_loss_weight, default_word_loss_weight, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # CRF module has parameters and it's forward function is not called in
        # model's forward function because of ONNX compatibility issue. This will
        # not work with DDP, thus setting find_unused_parameters to False to work
        # around, can be removed once DDP support params not used in model forward
        # function
        self.find_unused_parameters = False
        self.default_doc_loss_weight = default_doc_loss_weight
        self.default_word_loss_weight = default_word_loss_weight
        log_class_usage(__class__)

    @classmethod
    def create_embedding(cls, config, tensorizers):
        vocab = tensorizers["tokens"].vocab
        word_embedding = WordEmbedding(
            len(vocab),
            config.word_embedding.embed_dim,
            None,
            None,
            vocab.idx[SpecialTokens.UNK],
            [],
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

        return cls(
            config.default_doc_loss_weight,
            config.default_word_loss_weight,
            embedding,
            representation,
            decoder,
            output_layer,
        )

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens, _ = tensor_dict["tokens"]
        return (tokens, seq_lens)

    def arrange_targets(self, tensor_dict):
        intent_tensor = tensor_dict["doc_labels"]
        slot_tensor = tensor_dict["word_labels"]
        return intent_tensor, slot_tensor

    def vocab_to_export(self, tensorizers):
        return {"tokens_vals": list(tensorizers["tokens"].vocab)}

    def get_export_input_names(self, tensorizers):
        return ["tokens_vals", "tokens_lens"]

    def get_export_output_names(self, tensorizers):
        return ["doc_scores", "word_scores"]

    def arrange_model_context(self, tensor_dict):
        context = self.get_weights_context(tensor_dict)
        context["seq_lens"] = tensor_dict["tokens"][1]
        return context

    def get_weights_context(self, tensor_dict):
        batch_size = tensor_dict["doc_labels"].size()[0]
        return {
            "doc_weight": tensor_dict.get(
                "doc_weight",
                cuda_util.tensor(
                    [self.default_doc_loss_weight] * batch_size, dtype=torch.float
                ),
            ),
            "word_weight": tensor_dict.get(
                "word_weight",
                cuda_util.tensor(
                    [self.default_word_loss_weight] * batch_size, dtype=torch.float
                ),
            ),
        }

    def caffe2_export(self, tensorizers, tensor_dict, path, export_onnx_path=None):
        exporter = ModelExporter(
            ModelExporter.Config(),
            self.get_export_input_names(tensorizers),
            self.arrange_model_inputs(tensor_dict),
            self.vocab_to_export(tensorizers),
            self.get_export_output_names(tensorizers),
        )
        return exporter.export_to_caffe2(self, path, export_onnx_path=export_onnx_path)
