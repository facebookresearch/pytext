#!/usr/bin/env python3

from typing import List

import torch
from pytext.common.constants import DatasetFieldName
from pytext.config import ConfigBase
from pytext.config.contextual_intent_slot import ModelInput
from pytext.data import CommonMetadata
from pytext.models.decoders.joint_model_decoder import JointModelDecoder
from pytext.models.embeddings import EmbeddingList
from pytext.models.joint_model import JointModel
from pytext.models.model import Model
from pytext.models.module import create_module
from pytext.models.output_layer.intent_slot_output_layer import IntentSlotOutputLayer
from pytext.models.representations.contextual_intent_slot_rep import (
    ContextualIntentSlotRepresentation,
)


class ContextualIntentSlotModel(JointModel):
    """Contextual intent slot model
    The model takes contexts (sequences of texts), dictionary feature of the last
    text as input, calculates the doc and word representation with contexts
    embeddings and Then passes them to joint model decoder for intent and slot training.
    Used for e.g., dialog or long text.
    """

    class Config(JointModel.Config):
        representation: ContextualIntentSlotRepresentation.Config = ContextualIntentSlotRepresentation.Config()

    @classmethod
    def compose_embedding(cls, sub_embs):
        return EmbeddingList(
            embeddings=[
                EmbeddingList(
                    embeddings=[
                        sub_embs.get(ModelInput.TEXT),
                        sub_embs.get(ModelInput.DICT),
                        sub_embs.get(ModelInput.CHAR),
                        sub_embs.get(ModelInput.PRETRAINED),
                    ],
                    concat=True,
                ),
                sub_embs.get(ModelInput.SEQ),
            ],
            concat=False,
        )
