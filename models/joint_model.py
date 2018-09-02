#!/usr/bin/env python3
from typing import Union

from pytext.common.constants import DatasetFieldName
from pytext.config import ConfigBase
from pytext.config.component import create_module
from pytext.data import CommonMetadata
from pytext.models.model import Model

from .output_layer.intent_slot_output_layer import IntentSlotOutputLayer
from .projections.joint_model_projection import JointModelProjection
from .representations.jointblstm_rep import JointBLSTMRepresentation
from .representations.jointcnn_rep import JointCNNRepresentation


class JointModel(Model):
    class Config(ConfigBase):
        representation: Union[
            JointBLSTMRepresentation.Config, JointCNNRepresentation.Config
        ]
        output_config: IntentSlotOutputLayer.Config
        proj_config: JointModelProjection.Config = JointModelProjection.Config()
        default_doc_loss_weight: float = 0.2
        default_word_loss_weight: float = 0.5

    @classmethod
    def from_config(cls, model_config, feat_config, metadata: CommonMetadata):
        embedding = create_module(feat_config, metadata=metadata)
        representation = create_module(
            model_config.representation, embed_dim=embedding.embedding_dim
        )
        doc_class_num = metadata.labels[DatasetFieldName.DOC_LABEL_FIELD].vocab_size
        word_label_num = metadata.labels[DatasetFieldName.WORD_LABEL_FIELD].vocab_size
        projection = create_module(
            model_config.proj_config,
            from_dim_doc=representation.doc_representation_dim,
            from_dim_word=representation.word_representation_dim,
            to_dim_doc=doc_class_num,
            to_dim_word=word_label_num,
        )
        output_layer = create_module(model_config.output_config, metadata)
        return cls(embedding, representation, projection, output_layer)
