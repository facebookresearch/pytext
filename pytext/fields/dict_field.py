#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import Counter
from typing import List, Tuple

import torch
from pytext.common.constants import VocabMeta
from pytext.utils.data import no_tokenize
from torchtext import data as textdata, vocab

from .field import VocabUsingField


class DictFeatureField(VocabUsingField):
    dummy_model_input = (
        torch.tensor([[1], [1]], dtype=torch.long, device="cpu"),
        torch.tensor([[1.5], [2.5]], dtype=torch.float, device="cpu"),
        torch.tensor([[1], [1]], dtype=torch.long, device="cpu"),
    )

    def __init__(
        self,
        pad_token=VocabMeta.PAD_TOKEN,
        unk_token=VocabMeta.UNK_TOKEN,
        batch_first=True,
        **kwargs,
    ):
        super().__init__(
            sequential=True,
            batch_first=batch_first,
            tokenize=no_tokenize,
            use_vocab=True,
            pad_token=pad_token,
            unk_token=unk_token,
        )

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
            for x in data:
                if len(x) > 0:
                    counter.update(x[0])
        specials = [self.unk_token, self.pad_token]
        self.vocab = vocab.Vocab(counter, specials=specials, **kwargs)

    def pad(
        self, minibatch: List[Tuple[List[int], List[float], List[int]]]
    ) -> Tuple[List[List[int]], List[List[float]], List[int]]:
        # Pad a minibatch of dictionary features to be
        # batch_size * max_number_of_words * max_number_of_features
        # unpack the minibatch
        feats, weights, lengths = [], [], []
        for (fs, ws, ls) in minibatch:
            feats.append(fs)
            weights.append(ws)
            lengths.append(ls)

        lengths_flattened = [l for l_list in lengths for l in l_list]
        seq_lens = [len(l_list) for l_list in lengths]
        max_ex_len = self.pad_length(max(seq_lens))
        max_feat_len = max(lengths_flattened)
        all_lengths, all_feats, all_weights = [], [], []
        for i, seq_len in enumerate(seq_lens):
            ex_feats, ex_weights, ex_lengths = [], [], []
            feats_lengths, feats_vals, feats_weights = lengths[i], feats[i], weights[i]
            max_feat_len_example = max(feats_lengths)
            r_offset = 0
            for _ in feats_lengths:
                # The dict feats obtained from the featurizer will have necessary
                # padding at the utterance level. Therefore we move the offset by
                # max feature length in the example.
                ex_feats.extend(feats_vals[r_offset : r_offset + max_feat_len_example])
                ex_feats.extend(
                    [self.pad_token] * (max_feat_len - max_feat_len_example)
                )
                ex_weights.extend(
                    feats_weights[r_offset : r_offset + max_feat_len_example]
                )
                ex_weights.extend([0.0] * (max_feat_len - max_feat_len_example))
                r_offset += max_feat_len_example
            ex_lengths.extend(feats_lengths)
            # Pad examples
            ex_padding = (max_ex_len - seq_len) * max_feat_len
            ex_feats.extend([self.pad_token] * ex_padding)
            ex_weights.extend([0.0] * ex_padding)
            ex_lengths.extend([1] * (max_ex_len - seq_len))
            all_feats.append(ex_feats)
            all_weights.append(ex_weights)
            all_lengths.append(ex_lengths)
        return all_feats, all_weights, all_lengths

    def numericalize(self, arr, device=None):
        feats, weights, lengths = arr
        weights = torch.tensor(weights, dtype=torch.float, device=device)
        lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        feats = [[self.vocab.stoi[x] for x in ex] for ex in feats]
        feats = torch.tensor(feats, dtype=self.dtype, device=device)
        if not self.batch_first:
            arr.t_()
            weights.t_()
        feats = feats.contiguous()
        return feats, weights, lengths
