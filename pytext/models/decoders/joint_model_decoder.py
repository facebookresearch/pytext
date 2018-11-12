#!/usr/bin/env python3

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytext.config import ConfigBase

from .decoder_base import DecoderBase


class JointModelDecoder(DecoderBase):
    class Config(DecoderBase.Config):
        use_doc_probs_in_word: bool = False

    def __init__(
        self, config, from_dim_doc, from_dim_word, to_dim_doc, to_dim_word
    ) -> None:
        super().__init__(config)

        self.use_doc_probs_in_word = config.use_doc_probs_in_word
        self.out_d = nn.Linear(from_dim_doc, to_dim_doc)

        if self.use_doc_probs_in_word:
            from_dim_word += to_dim_doc

        self.out_w = nn.Linear(from_dim_word, to_dim_word)

    def forward(self, x_d: torch.Tensor, x_w: torch.Tensor) -> List[torch.Tensor]:
        logit_d = self.out_d(x_d)
        if self.use_doc_probs_in_word:
            # Get doc probability distribution
            doc_prob = F.softmax(logit_d, 1)
            word_input_shape = x_w.size()
            doc_prob = doc_prob.unsqueeze(1).repeat(1, word_input_shape[1], 1)
            x_w = torch.cat((x_w, doc_prob), 2)

        return [logit_d, self.out_w(x_w)]

    def get_decoder(self) -> List[nn.Module]:
        return [self.out_d, self.out_w]
