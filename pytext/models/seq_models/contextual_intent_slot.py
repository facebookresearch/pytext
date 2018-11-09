#!/usr/bin/env python3

from typing import List

import torch
from pytext.common.constants import DatasetFieldName
from pytext.config import ConfigBase
from pytext.models.module import create_module
from pytext.data import CommonMetadata
from pytext.models.decoders.joint_model_decoder import JointModelDecoder
from pytext.models.model import DataParallelModel, Model
from pytext.models.output_layer.intent_slot_output_layer import IntentSlotOutputLayer
from pytext.models.representations.contextual_intent_slot_rep import (
    ContextualIntentSlotRepresentation,
)
from pytext.utils import cuda_utils


class ContextualIntentSlotModel(Model):
    """Contextual intent slot model
    The model takes contexts (sequences of texts), dictionary feature of the last
    text as input, calculates the doc and word representation with contexts
    embeddings and Then passes them to joint model decoder for intent and slot training.
    Used for e.g., dialog or long text.
    """

    class Config(ConfigBase):
        representation: ContextualIntentSlotRepresentation.Config = ContextualIntentSlotRepresentation.Config()
        output_layer: IntentSlotOutputLayer.Config = (IntentSlotOutputLayer.Config())
        decoder: JointModelDecoder.Config = JointModelDecoder.Config()
        default_doc_loss_weight: float = 0.2
        default_word_loss_weight: float = 0.5

    @classmethod
    def from_config(cls, model_config, feat_config, metadata: CommonMetadata):
        embedding = create_module(feat_config, metadata=metadata)
        if embedding.seq_word_embedding_dim == 0:
            raise ValueError("Sequence word level embedding config must be provided.")
        representation = create_module(
            model_config.representation,
            embed_dim=embedding.embedding_dim,
            seq_embed_dim=embedding.seq_word_embedding_dim,
        )
        doc_class_num = metadata.labels[DatasetFieldName.DOC_LABEL_FIELD].vocab_size
        word_label_num = metadata.labels[DatasetFieldName.WORD_LABEL_FIELD].vocab_size
        decoder = create_module(
            model_config.decoder,
            from_dim_doc=representation.joint_rep.doc_representation_dim,
            from_dim_word=representation.joint_rep.word_representation_dim,
            to_dim_doc=doc_class_num,
            to_dim_word=word_label_num,
        )
        output_layer = create_module(model_config.output_layer, metadata)
        return cls(embedding, representation, decoder, output_layer)

    def forward(
        self,
        tokens,
        seq_lens,
        dict_feat,
        chars,
        pretrained_model_embedding,
        seq_feat,
    ) -> List[torch.Tensor]:
        seq_tokens, sen_seq_lens, _ = seq_feat
        word_embed, seq_embed = self.embedding(
            tokens,
            seq_lens,
            dict_feat,
            chars,
            pretrained_model_embedding,
            seq_tokens,
        )
        return cuda_utils.parallelize(
            DataParallelModel(self.representation, self.decoder),
            (word_embed, seq_embed, seq_lens, sen_seq_lens),
        )  # returned Tensor's dim = (batch_size, num_classes)
