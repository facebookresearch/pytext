#!/usr/bin/env python3
from typing import Union

from pytext.common.constants import DatasetFieldName
from pytext.config import ConfigBase
from pytext.data import CommonMetadata
from pytext.models.model import Model
from pytext.models.module import create_module

from .decoders import IntentSlotModelDecoder
from .output_layer.intent_slot_output_layer import IntentSlotOutputLayer
from .representations.bilstm_doc_slot_attention import BiLSTMDocSlotAttention
from .representations.jointcnn_rep import JointCNNRepresentation


class JointModel(Model):
    class Config(ConfigBase):
        representation: Union[
            BiLSTMDocSlotAttention.Config, JointCNNRepresentation.Config
        ] = BiLSTMDocSlotAttention.Config()
        output_layer: IntentSlotOutputLayer.Config = (IntentSlotOutputLayer.Config())
        decoder: IntentSlotModelDecoder.Config = IntentSlotModelDecoder.Config()
        default_doc_loss_weight: float = 0.2
        default_word_loss_weight: float = 0.5

    @classmethod
    def from_config(cls, model_config, feat_config, metadata: CommonMetadata):
        embedding = cls.create_embedding(feat_config, metadata)
        representation = create_module(
            model_config.representation, embed_dim=embedding.embedding_dim
        )
        doc_label_meta, word_label_meta = metadata.target
        decoder = create_module(
            model_config.decoder,
            in_dim_doc=representation.doc_representation_dim,
            in_dim_word=representation.word_representation_dim,
            out_dim_doc=doc_label_meta.vocab_size,
            out_dim_word=word_label_meta.vocab_size,
        )
        output_layer = create_module(
            model_config.output_layer, doc_label_meta, word_label_meta
        )
        return cls(embedding, representation, decoder, output_layer)
