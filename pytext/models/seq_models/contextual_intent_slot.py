#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional

from pytext.data.tensorizers import SeqTokenTensorizer
from pytext.models.embeddings import EmbeddingList, WordEmbedding
from pytext.models.joint_model import IntentSlotModel
from pytext.models.module import create_module
from pytext.models.representations.contextual_intent_slot_rep import (
    ContextualIntentSlotRepresentation,
)


class ContextualIntentSlotModel(IntentSlotModel):
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

    class Config(IntentSlotModel.Config):
        class ModelInput(IntentSlotModel.Config.ModelInput):

            seq_tokens: Optional[
                SeqTokenTensorizer.Config
            ] = SeqTokenTensorizer.Config()

        inputs: ModelInput = ModelInput()
        seq_embedding: Optional[WordEmbedding.Config] = WordEmbedding.Config()
        representation: ContextualIntentSlotRepresentation.Config = ContextualIntentSlotRepresentation.Config()

    @classmethod
    def create_embedding(cls, config, tensorizers):
        word_emb = create_module(
            config.word_embedding,
            tensorizer=tensorizers["tokens"],
            init_from_saved_state=config.init_from_saved_state,
        )

        seq_emb_tensorizer = tensorizers["seq_tokens"]
        seq_emb = create_module(config.seq_embedding, tensorizer=seq_emb_tensorizer)
        return EmbeddingList(
            [EmbeddingList([word_emb], concat=True), seq_emb], concat=False
        )

    def vocab_to_export(self, tensorizers):
        return {
            "tokens_vals": list(tensorizers["tokens"].vocab),
            "seq_tokens_vals": list(tensorizers["seq_tokens"].vocab),
        }

    def get_export_input_names(self, tensorizers):
        return ["tokens_vals", "seq_tokens_vals", "tokens_lens", "seq_tokens_lens"]

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens, _ = tensor_dict["tokens"]
        arranged_inputs = [tokens]
        seq_emb_inputs, seq_word_lens = tensor_dict.get("seq_tokens")
        arranged_inputs.append(seq_emb_inputs)
        arranged_inputs.append(seq_lens)
        arranged_inputs.append(seq_word_lens)
        return tuple(arranged_inputs)
