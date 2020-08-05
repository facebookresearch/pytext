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
        loss: Optional[Loss] = None,
    ):
        super().__init__()
        self.word_embedding = word_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.head = SequenceClassificationHead(loss)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        tokens = inputs["token_ids"]
        # denses = inputs["dense"] if "dense" in inputs else None
        batch_array = torch.tensor([])
        word_embedding_output = self.word_embedding(tokens)
        for sentence in word_embedding_output:
            sentence_array = torch.tensor([])
            for word in sentence:
                word = torch.tensor([[list(word)]])
                encoder_output = self.encoder(word)
                decoder_output = self.decoder(encoder_output)
                sentence_array = torch.cat((sentence_array, decoder_output), 0)
            batch_array = torch.cat((batch_array, sentence_array), 0)
        return batch_array

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
    decoder_act,
    num_slots,
    vocab,
):
    Embedder = WordEmbedding(pretrain_embed, vocab, embed_dim)
    Encoder = DocNNEncoder(embed_dim, kernel_num, kernel_sizes, dropout)
    Decoder = MLPDecoder(
        Encoder.out_dim, num_slots, bias, decoder_hidden_dims, decoder_act
    )
    SlotLabellingOutput = IntentSlotLabellingModel(Embedder, Encoder, Decoder)
    return SlotLabellingOutput


def build_dumb_slot_labelling_model():
    return build_slot_labelling_model(
        None,
        5,
        100,
        [10 for i in range(100)],
        0.4,
        False,
        None,
        None,
        5,
        Vocabulary([SpecialTokens.UNK, SpecialTokens.PAD, "the", "cat"]),
    )
