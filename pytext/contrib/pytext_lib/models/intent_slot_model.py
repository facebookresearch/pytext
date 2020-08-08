#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, Optional, Tuple, Union  # nora

import torch
import torch.nn as nn
from pytext.common.constants import SpecialTokens
from pytext.data.utils import Vocabulary

from .classification_heads import ClassificationHead
from .doc_model import DocNNEncoder, WordEmbedding
from .mlp_decoder import MLPDecoder
from .roberta import RoBERTaEncoder


class IntentSlotLabellingModel(nn.Module):
    def __init__(
        self,
        slot_decoder: MLPDecoder,
        slot_encoder: Optional[Union[DocNNEncoder, RoBERTaEncoder]] = None,
    ):
        super().__init__()
        self.slot_decoder = slot_decoder
        self.slot_encoder = slot_encoder
        self.head = ClassificationHead(is_binary=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.slot_decoder(inputs)
        return logits

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        all_word_encodings = torch.tensor([])  # 3d
        for sentence in embeddings:
            curr_sentence_words = torch.tensor([])
            for word in sentence:
                word = torch.tensor([[list(word)]])
                encoded_word = self.slot_encoder(word)
                curr_sentence_words = torch.cat((curr_sentence_words, encoded_word), 0)
            all_word_encodings = torch.cat(
                (all_word_encodings, curr_sentence_words.unsqueeze(0)), 0
            )
        return all_word_encodings

    def get_results(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.head(logits)

    def get_loss(self, logits: torch.Tensor, targets) -> torch.Tensor:
        return self.head.get_loss(logits, targets)


class IntentDocLabellingModel(nn.Module):
    def __init__(
        self,
        doc_decoder: MLPDecoder,
        doc_encoder: Optional[Union[DocNNEncoder, RoBERTaEncoder]] = None,
    ):
        super().__init__()
        self.doc_decoder = doc_decoder
        self.doc_encoder = doc_encoder
        self.head = ClassificationHead(is_binary=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.doc_decoder(inputs)
        return logits

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        all_sentence_encodings = torch.tensor([])  # 2d
        for sentence in embeddings:
            encoded_sentence = self.doc_encoder(sentence.unsqueeze(0))
            all_sentence_encodings = torch.cat(
                (all_sentence_encodings, encoded_sentence), 0
            )
        return all_sentence_encodings

    def get_results(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.head(logits)

    def get_loss(self, logits: torch.Tensor, targets) -> torch.Tensor:
        return self.head.get_loss(logits, targets)


class IntentSlotJointModel(nn.Module):
    def __init__(
        self,
        word_embedding: WordEmbedding,
        slot_output: IntentSlotLabellingModel,
        doc_output: IntentDocLabellingModel,
        doc_weight: float = 0.4,
        use_intent: bool = True,
        len_intent: int = 0,
    ):
        super().__init__()
        self.word_embedding = word_embedding
        self.use_intent = use_intent
        self.len_intent = len_intent
        self.slot_output = slot_output
        self.doc_output = doc_output
        self.doc_weight = doc_weight

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        all_intents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = inputs["token_ids"]
        slots_batch = torch.tensor([])
        intent_batch = torch.tensor([])
        word_embedding_output = self.word_embedding(tokens)
        sentence_encodings = self.doc_output.encode(word_embedding_output)
        word_encodings = self.slot_output.encode(word_embedding_output)
        for sentence_i in range(len(sentence_encodings)):
            encoded_sentence = sentence_encodings[sentence_i]
            doc_output = self.doc_output(encoded_sentence).unsqueeze(0)  # 2d
            intent_batch = torch.cat((intent_batch, doc_output), 0)  # 2d
            if all_intents is not None:
                doc_pred = all_intents[sentence_i]
            else:
                doc_pred = self.doc_output.get_results(doc_output)[0]

            curr_sentence_words = word_encodings[sentence_i]
            sentence_slot_tensor = torch.tensor([])
            for word_encoding in curr_sentence_words:
                word_decoder_input = word_encoding.unsqueeze(0)
                if self.use_intent:
                    add_feats = self.prep_add_feats(doc_pred)
                    word_decoder_input = torch.cat((word_decoder_input, add_feats), 1)
                slot_output = self.slot_output(word_decoder_input)
                sentence_slot_tensor = torch.cat(
                    (sentence_slot_tensor, slot_output), 0
                )  # 2d
            slots_batch = torch.cat((slots_batch, sentence_slot_tensor), 0)  # 2d
        return intent_batch, slots_batch

    def get_preds(
        self, logits: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        doc_preds = self.doc_output.get_results(logits[0])[0]
        slot_preds = self.slot_output.get_results(logits[1])[0]
        return doc_preds, slot_preds

    def get_loss(
        self,
        logits: Tuple[torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor],
    ):
        doc_loss = self.doc_output.get_loss(logits[0], targets[0])
        word_loss = self.slot_output.get_loss(logits[1], targets[1])
        loss = self.doc_weight * doc_loss + (1 - self.doc_weight) * word_loss
        return loss

    def prep_add_feats(self, intent: Optional[int] = 0):
        add_feats = None
        if self.use_intent:
            intent_encoding = [0.0 for i in range(self.len_intent)]
            intent_encoding[intent] = 1.0
            add_feats = torch.tensor([intent_encoding])
        return add_feats


def build_intent_joint_model(
    use_intent,
    loss_doc_weight,
    pretrain_embed,
    embed_dim,
    slot_kernel_num,
    slot_kernel_sizes,
    doc_kernel_num,
    doc_kernel_sizes,
    slot_bias,
    slot_decoder_hidden_dims,
    doc_bias,
    doc_decoder_hidden_dims,
    num_slots,
    num_intents,
    vocab,
    add_feat_len=0,
):
    embedder = WordEmbedding(pretrain_embed, vocab, embed_dim)
    slot_encoder = DocNNEncoder(embed_dim, slot_kernel_num, slot_kernel_sizes)
    doc_encoder = DocNNEncoder(embed_dim, doc_kernel_num, doc_kernel_sizes)
    slot_decoder = MLPDecoder(
        slot_encoder.out_dim + add_feat_len,
        num_slots,
        slot_bias,
        slot_decoder_hidden_dims,
    )
    doc_decoder = MLPDecoder(
        doc_encoder.out_dim, num_intents, doc_bias, doc_decoder_hidden_dims
    )
    slot_output = IntentSlotLabellingModel(slot_decoder, slot_encoder)
    doc_output = IntentDocLabellingModel(doc_decoder, doc_encoder)
    joint_model = IntentSlotJointModel(
        embedder,
        slot_output,
        doc_output,
        loss_doc_weight,
        use_intent,
        len_intent=num_intents,
    )
    return joint_model


def build_dumb_intent_slot_model():
    return build_intent_joint_model(
        use_intent=False,
        loss_doc_weight=0.4,
        pretrain_embed=None,
        embed_dim=10,
        slot_kernel_num=10,
        slot_kernel_sizes=[10 for i in range(100)],
        doc_kernel_num=10,
        doc_kernel_sizes=[10 for i in range(100)],
        slot_bias=True,
        slot_decoder_hidden_dims=None,
        doc_bias=True,
        doc_decoder_hidden_dims=None,
        num_slots=26,
        num_intents=43,
        vocab=Vocabulary([SpecialTokens.UNK, SpecialTokens.PAD, "the", "cat"]),
        add_feat_len=0,
    )
