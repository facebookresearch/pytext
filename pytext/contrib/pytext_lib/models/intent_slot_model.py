#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights ReservedModel

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional  # nora

import torch
import torch.nn as nn
from pytext.common.constants import SpecialTokens
from pytext.data.utils import Vocabulary
from pytext.loss.loss import Loss

from .classification_heads import SequenceClassificationHead
from .doc_model import DocNNEncoder, WordEmbedding
from .mlp_decoder import MLPDecoder


class IntentSlotLabellingModel(nn.Module):
    def __init__(
        self,
        word_embedding: WordEmbedding,
        # encoder must take len(text) inputs
        encoder: DocNNEncoder,
        # encoder must output the same as input of decoder
        decoder: MLPDecoder,
        # must output len(text)*len(possible slots),
        use_intent: bool = False,
        len_intent: int = 0,
        loss: Optional[Loss] = None,
    ):
        super().__init__()
        self.word_embedding = word_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.head = SequenceClassificationHead(loss)
        self.use_intent = use_intent
        self.len_intent = len_intent

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        all_intents: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        tokens = inputs["token_ids"]
        # denses = inputs["dense"] if "dense" in inputs else None
        batch_array = torch.tensor([])
        word_embedding_output = self.word_embedding(tokens)  # 3d tensor
        for sentence_index in range(len(word_embedding_output)):
            sentence = word_embedding_output[sentence_index]
            sentence_array = torch.tensor([])
            for word in sentence:
                word = torch.tensor([[list(word)]])  # 3d tensor
                encoder_output = self.encoder(word)  # 2d tensor
                decoder_input = encoder_output
                if self.use_intent and all_intents is not None:
                    intent = all_intents[sentence_index]
                    add_feats = self.prep_add_feats(intent)
                    decoder_input = torch.cat((encoder_output, add_feats), 1)
                decoder_output = self.decoder(decoder_input)  # 2d tensor
                sentence_array = torch.cat(
                    (sentence_array, decoder_output), 0
                )  # 2d tensor
            batch_array = torch.cat((batch_array, sentence_array), 0)  # 2d tensor
        return batch_array

    def prep_add_feats(self, intent: Optional[int] = 0):
        add_feats = None
        if self.use_intent:
            intent_encoding = [0.0 for i in range(self.len_intent)]
            intent_encoding[intent] = 1.0
            add_feats = torch.tensor([intent_encoding])
        return add_feats

    def get_pred(self, logits: List[torch.Tensor]) -> torch.Tensor:
        return self.head(logits)

    def get_loss(self, logits: List[torch.Tensor], targets) -> torch.Tensor:
        return self.head.get_loss(logits, targets)


def build_slot_labelling_model(
    pretrain_embed,
    embed_dim,
    kernel_num,
    kernel_sizes,
    dropout,
    bias,
    decoder_hidden_dims,
    num_slots,
    num_intents,
    vocab,
):
    Embedder = WordEmbedding(pretrain_embed, vocab, embed_dim)
    Encoder = DocNNEncoder(embed_dim, kernel_num, kernel_sizes, dropout)
    Decoder = MLPDecoder(Encoder.out_dim, num_slots, bias, decoder_hidden_dims)
    SlotLabellingModel = IntentSlotLabellingModel(
        Embedder, Encoder, Decoder, use_intent=False, len_intent=num_intents
    )
    return SlotLabellingModel


def build_dumb_slot_labelling_model():
    return build_slot_labelling_model(
        None,
        5,
        10,
        [10 for i in range(10)],
        0.4,
        False,
        None,
        5,
        5,
        Vocabulary([SpecialTokens.UNK, SpecialTokens.PAD, "the", "cat"]),
    )
