#!/usr/bin/env python3
from collections import Counter
from itertools import chain
from typing import Any, Counter as CounterType, List

from torchtext import data as textdata, vocab


class TextField(textdata.Field):
    def __init__(self, **kwargs):
        super(TextField, self).__init__(**kwargs)

    def build_vocab(self, *args, existing_vocab: List[str] = None, **kwargs):
        """Construct the Vocab object for this field frsom one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter: CounterType[str] = Counter()
        sources: List[Any] = []
        for arg in args:
            if isinstance(arg, textdata.Dataset):
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
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))

        specials: List[str] = []
        if existing_vocab:
            specials = existing_vocab
        else:
            specials = [
                tok
                for tok in [
                    self.unk_token,
                    self.pad_token,
                    self.init_token,
                    self.eos_token,
                ]
                if tok is not None
            ]

        self.vocab = vocab.Vocab(counter, specials=specials, **kwargs)
