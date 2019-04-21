#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import List

import torch
from pytext.utils import data

from .field import Field, TextFeatureField


class ContextualTokenEmbeddingField(Field):
    def __init__(self, **kwargs):
        super().__init__(
            sequential=True,
            use_vocab=False,
            batch_first=True,
            tokenize=data.no_tokenize,
            dtype=torch.float,
            unk_token=None,
            pad_token=None,
        )
        batch_size = TextFeatureField.dummy_model_input.size(0)
        num_tokens = TextFeatureField.dummy_model_input.size(1)
        embed_dim = kwargs.get("embed_dim", 0)
        self.dummy_model_input = torch.tensor(
            [[1.0] * embed_dim * num_tokens] * batch_size,
            dtype=torch.float,
            device="cpu",
        )

    def pad(self, minibatch: List[List[List[float]]]) -> List[List[List[float]]]:
        """
        Example of padded minibatch:
        ::

            [[[0.1, 0.2, 0.3, 0.4, 0.5],
              [1.1, 1.2, 1.3, 1.4, 1.5],
              [2.1, 2.2, 2.3, 2.4, 2.5],
              [3.1, 3.2, 3.3, 3.4, 3.5],
             ],
             [[0.1, 0.2, 0.3, 0.4, 0.5],
              [1.1, 1.2, 1.3, 1.4, 1.5],
              [2.1, 2.2, 2.3, 2.4, 2.5],
              [0.0, 0.0, 0.0, 0.0, 0.0],
             ],
             [[0.1, 0.2, 0.3, 0.4, 0.5],
              [1.1, 1.2, 1.3, 1.4, 1.5],
              [0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0],
             ],
            ]
        """
        padded_minibatch = copy.deepcopy(minibatch)

        max_sentence_length, word_embedding_dim = 0, 0
        for sent in padded_minibatch:
            max_sentence_length = max(max_sentence_length, len(sent))
            j = 0
            while j < len(sent) and word_embedding_dim == 0:
                word_embedding_dim = len(sent[j])
                j += 1
        max_sentence_length = self.pad_length(max_sentence_length)

        for i, sentence in enumerate(padded_minibatch):
            if len(sentence) < max_sentence_length:
                one_word_embedding = [0.0] * word_embedding_dim
                padding = [one_word_embedding] * (max_sentence_length - len(sentence))
                padded_minibatch[i].extend(padding)
        return padded_minibatch

    def numericalize(self, batch, device=None):
        return (
            torch.tensor(batch, dtype=self.dtype, device=device)
            .contiguous()
            .view(-1, len(batch[0]) * len(batch[0][0]))
        )
