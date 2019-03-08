#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from collections import Counter
from typing import List

import torch
from pytext.common.constants import VocabMeta
from pytext.utils.data import no_tokenize
from torchtext import data as textdata, vocab

from .field import VocabUsingField


class CharFeatureField(VocabUsingField):
    dummy_model_input = torch.tensor(
        [[[1, 1, 1]], [[1, 1, 1]]], dtype=torch.long, device="cpu"
    )

    def __init__(
        self,
        pad_token=VocabMeta.PAD_TOKEN,
        unk_token=VocabMeta.UNK_TOKEN,
        batch_first=True,
        max_word_length=20,
        min_freq=1,
        **kwargs
    ):
        super().__init__(
            sequential=True,  # Otherwise pad is set to None in textdata.Field
            batch_first=batch_first,
            tokenize=no_tokenize,
            use_vocab=True,
            pad_token=pad_token,
            unk_token=unk_token,
            min_freq=min_freq,
        )
        self.max_word_length = self.pad_length(max_word_length)

    def build_vocab(self, *args, **kwargs):
        sources = []
        for arg in args:
            if isinstance(arg, textdata.Dataset):
                sources += [
                    getattr(arg, name)
                    for name, field in arg.fields.items()
                    if field is self
                ]
            else:
                sources.append(arg)

        counter = Counter()
        for data in sources:
            # data is the return value of preprocess().
            for sentence in data:
                for word_chars in sentence:
                    # update treats word as an iterable, so this will add all
                    # the characters from the word, not the word itself.
                    counter.update(word_chars)
        specials = [self.unk_token, self.pad_token]

        self.vocab = vocab.Vocab(counter, specials=specials, **kwargs)

    def pad(self, minibatch: List[List[List[str]]]) -> List[List[List[str]]]:
        """
        Example of minibatch:
        ::

            [[['p', 'l', 'a', 'y', '<PAD>', '<PAD>'],
              ['t', 'h', 'a', 't', '<PAD>', '<PAD>'],
              ['t', 'r', 'a', 'c', 'k', '<PAD>'],
              ['o', 'n', '<PAD>', '<PAD>', '<PAD>', '<PAD>'],
              ['r', 'e', 'p', 'e', 'a', 't']
             ], ...
            ]
        """
        # If we change the same minibatch object then the underlying data
        # will get corrupted. Hence deep copy the minibatch object.
        padded_minibatch = copy.deepcopy(minibatch)

        max_sentence_length = self.pad_length(max(len(sent) for sent in minibatch))
        max_word_length = self.pad_length(
            max(len(word) for sent in minibatch for word in sent)
        )
        max_word_length = min(self.max_word_length, max_word_length)

        for i, sentence in enumerate(minibatch):
            for j, word in enumerate(sentence):
                char_padding = [self.pad_token] * (max_word_length - len(word))
                padded_minibatch[i][j].extend(char_padding)
                padded_minibatch[i][j] = padded_minibatch[i][j][:max_word_length]
            if len(sentence) < max_sentence_length:
                for _ in range(max_sentence_length - len(sentence)):
                    char_padding = [self.pad_token] * max_word_length
                    padded_minibatch[i].append(char_padding)

        return padded_minibatch

    def numericalize(self, batch, device=None):
        batch_char_ids = []
        for sentence in batch:
            sentence_char_ids = super().numericalize(sentence, device=device)
            batch_char_ids.append(sentence_char_ids)
        return torch.stack(batch_char_ids, dim=0)
