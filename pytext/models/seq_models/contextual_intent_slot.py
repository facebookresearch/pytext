#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.config.contextual_intent_slot import ModelInput
from pytext.models.embeddings import EmbeddingList
from pytext.models.joint_model import JointModel
from pytext.models.representations.contextual_intent_slot_rep import (
    ContextualIntentSlotRepresentation,
)


class ContextualIntentSlotModel(JointModel):
    """
    Joint Model for Intent classification and slot tagging with inputs of contextual
    information (sequence of utterances) and dictionary feature of the last utterance.

    Training data should include:
    doc_label (string): intent classification label of either the sequence of \
    utterances or just the last sentence
    word_label (string): slot tagging label of the last utterance in the format\
    of start_idx:end_idx:slot_label, multiple slots are separated by a comma
    text (list of string): sequence of utterances for training
    dict_feat (dict): a dict of features that contains the feature of each word\
    in the last utterance

    Following is an example of raw columns from training data:

    ==========  =======================================================================
    doc_label   reply-where
    word_label  10:20:restaurant_name
    text        ["dinner at 6?", "wanna try Tomi Sushi?"]
    dict_feat   {"tokenFeatList": [{"tokenIdx": 2, "features": {"poi:eatery": 0.66}},
                                   {"tokenIdx": 3, "features": {"poi:eatery": 0.66}}]}
    ==========  =======================================================================

    """

    class Config(JointModel.Config):
        representation: ContextualIntentSlotRepresentation.Config = ContextualIntentSlotRepresentation.Config()

    @classmethod
    def compose_embedding(cls, sub_embs, metadata):
        """Compose embedding list for ContextualIntentSlot model training.
        The first is the word embedding of the last utterance concatenated with the
        word level dictionary feature. The second is the word embedding of a
        sequence of utterances (includes the last utterance). Two embeddings are
        not concatenated and passed to the model individually.

        Args:
            sub_embs (type): sub-embeddings.

        Returns:
            type: EmbeddingList object contains embedding of the last utterance with
                dictionary feature and embedding of the sequence of utterances.

        """
        return EmbeddingList(
            embeddings=[
                EmbeddingList(
                    embeddings=[
                        sub_embs.get(ModelInput.TEXT),
                        sub_embs.get(ModelInput.DICT),
                        sub_embs.get(ModelInput.CHAR),
                        sub_embs.get(ModelInput.CONTEXTUAL_TOKEN_EMBEDDING),
                    ],
                    concat=True,
                ),
                sub_embs.get(ModelInput.SEQ),
            ],
            concat=False,
        )
