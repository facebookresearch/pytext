#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from collections import Counter, OrderedDict
from itertools import chain
from typing import List, Optional, Tuple, Union

import six
import torch
from pytext.common.constants import VocabMeta
from pytext.fields import TextFeatureField
from pytext.utils.data import is_number, unkify
from torchtext.data import Dataset


class TextFeatureFieldWithSpecialUnk(TextFeatureField):
    def __init__(self, *args, unkify_func=unkify, **kwargs):
        super().__init__(*args, **kwargs)
        self.unkify_func = unkify_func
        self.unk_num_token = VocabMeta.UNK_NUM_TOKEN

    def build_vocab(self, *args, min_freq=1, **kwargs):
        """
        Code is exactly same as as torchtext.data.Field.build_vocab() before the
        UNKification logic. The reason super().build_vocab() cannot be called is
        because the Counter object computed in torchtext.data.Field.build_vocab()
        is required for UNKification and, that object cannot be recovered after
        super().build_vocab() call is made.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [
                    getattr(arg, name)
                    for name, field in arg.fields.items()
                    if field is self
                ]
            else:
                sources.append(arg)

        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                x = [item for item in x if not is_number(item)]
                # All numbers are mapped to self.unk_num_token
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(
            OrderedDict.fromkeys(
                tok
                for tok in [
                    self.unk_token,
                    self.pad_token,
                    self.init_token,
                    self.eos_token,
                    self.unk_num_token,
                ]
                if tok is not None
            )
        )

        # Special UNKification logic.
        if self.unkify_func:
            new_counter = Counter()
            for item in counter:
                new_item = item
                if counter[item] < min_freq:
                    new_item = self.unkify_func(item)
                new_counter.update([new_item] * counter[item])
            counter = new_counter

        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def numericalize(
        self,
        arr: Union[List[List[str]], Tuple[List[List[str]], List[int]]],
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Code is exactly same as torchtext.data.Field.numericalize() except the
        call to self._get_idx(x) instead of self.vocab.stoi[x] for getting the
        index of an item from vocab.
        This is needed because torchtext doesn't allow custom UNKification.
        So, TextFeatureFieldWithSpecialUnk field's constructor accepts a function
        unkify_func() that can be used to UNKifying instead of assigning all UNKs
        a default value.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError(
                "Field has include_lengths set to True, but "
                "input data is not a tuple of "
                "(data batch, batch lengths)."
            )
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            if self.sequential:
                arr = [[self._get_idx(x) for x in ex] for ex in arr]
            else:
                arr = [self._get_idx(x) for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype)
                )
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explicitly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [
                    numericalization_func(x) if isinstance(x, six.string_types) else x
                    for x in arr
                ]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        var = torch.tensor(arr, dtype=self.dtype, device=device)

        if self.sequential and not self.batch_first:
            var.t_()
        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def _get_idx(self, item):
        if item in self.vocab.stoi:
            return self.vocab.stoi[item]
        else:
            return self.vocab.stoi[self.unkify_func(item)]
