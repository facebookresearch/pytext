#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder_base import DecoderBase


class IntentSlotModelDecoder(DecoderBase):
    """
    `IntentSlotModelDecoder` implements the decoder layer for intent-slot models.
    Intent-slot models jointly predict intent and slots from an utterance.
    At the core these models learn to jointly perform document classification
    and word tagging tasks.

    `IntentSlotModelDecoder` accepts arguments for decoding both document
     classification and word tagging tasks, namely, `in_dim_doc` and `in_dim_word`.

    Args:
        config (type): Configuration object of type IntentSlotModelDecoder.Config.
        in_dim_doc (type): Dimension of input Tensor for projecting document
        representation.
        in_dim_word (type): Dimension of input Tensor for projecting word
        representation.
        out_dim_doc (type): Dimension of projected output Tensor for document
        classification.
        out_dim_word (type): Dimension of projected output Tensor for word tagging.

    Attributes:
        use_doc_probs_in_word (bool): Whether to use intent probabilities for
        predicting slots.
        doc_decoder (type): Document/intent decoder module.
        word_decoder (type): Word/slot decoder module.

    """

    class Config(DecoderBase.Config):
        """
        Configuration class for `IntentSlotModelDecoder`.

        Attributes:
            use_doc_probs_in_word (bool): Whether to use intent probabilities
                for predicting slots.
        """

        use_doc_probs_in_word: bool = False

    def __init__(
        self,
        config: Config,
        in_dim_doc: int,
        in_dim_word: int,
        out_dim_doc: int,
        out_dim_word: int,
    ) -> None:
        super().__init__(config)

        self.use_doc_probs_in_word = config.use_doc_probs_in_word
        self.doc_decoder = nn.Linear(in_dim_doc, out_dim_doc)

        if self.use_doc_probs_in_word:
            in_dim_word += out_dim_doc

        self.word_decoder = nn.Linear(in_dim_word, out_dim_word)

    def forward(
        self, x_d: torch.Tensor, x_w: torch.Tensor, dense: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        if dense is not None:
            logit_d = self.doc_decoder(torch.cat((x_d, dense), 1))
        else:
            logit_d = self.doc_decoder(x_d)

        if self.use_doc_probs_in_word:
            # Get doc probability distribution
            doc_prob = F.softmax(logit_d, 1)
            word_input_shape = x_w.size()
            doc_prob = doc_prob.unsqueeze(1).repeat(1, word_input_shape[1], 1)
            x_w = torch.cat((x_w, doc_prob), 2)

        if dense is not None:
            word_input_shape = x_w.size()
            dense = dense.unsqueeze(1).repeat(1, word_input_shape[1], 1)
            x_w = torch.cat((x_w, dense), 2)

        return [logit_d, self.word_decoder(x_w)]

    def get_decoder(self) -> List[nn.Module]:
        """Returns the document and word decoder modules.
        """
        return [self.doc_decoder, self.word_decoder]
