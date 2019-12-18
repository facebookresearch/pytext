#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import io
from typing import Dict, List

import torch
from pytext.torchscript.utils import utf8_chars
from pytext.utils.file_io import PathManager


class ScriptBPE(torch.jit.ScriptModule):
    """Byte-pair encoding implementation in TorchScript.

    vocab_file should be a file-like object separated by newlines, where each line
    consists of a word and a count separated by whitespace. Words in the vocab
    therefore can't contain space (according to python regex \\s). The vocab file
    should be sorted according to the importance of each token, and they will be
    merged in this priority; the actual score values are irrelevant.

    eow_token should be a string that is appended to the last character and token,
    and that token is used at each step in the process and returned at the end.
    You should set this to be consistent with the EOW signature used however you
    generated your ScriptBPE vocab file.

    >>> import io
    >>> vocab_file = io.StringIO('''
    hello_EOW 20
    world_EOW 18
    th  17
    is_EOW 16
    bpe_EOW 15
    ! 14
    h 13
    t 6
    s_EOW 2
    i -1
    ii -2
    ''')
    >>> bpe = ScriptBPE.from_vocab_file(vocab_file)
    >>> bpe.tokenize(["hello", "world", "this", "is", "bpe"])
    ["hello_EOW", "world_EOW", "th", "is_EOW", "is_EOW", "bpe_EOW"]
    >>> bpe.tokenize(["iiiis"])
    ["ii", "i", "is_EOW"]

    """

    def __init__(self, vocab: Dict[str, int], eow: str = "_EOW"):
        """vocab is a dictionary from BPE segments, including any EOW elements,
        to their priority in joining. Priority must be an integer, should not be
        negative, and should not contain ties. In the case of negative priorities,
        segments with negative priorities will be ignored. In the case of ties,
        ties will be broken according to left-to-right byte order precedence, but
        this behavior isn't guaranteed and may change in the future.

        eow should be a string which corresponds to the EOW used in the vocab
        dictionary."""
        super().__init__()
        self.vocab = torch.jit.Attribute(vocab, Dict[str, int])
        self.eow = torch.jit.Attribute(eow, str)

    @classmethod
    def from_vocab_file(cls, vocab_file: io.IOBase) -> "ScriptBPE":
        return cls(cls.load_vocab(vocab_file))

    @classmethod
    def from_vocab_filename(cls, vocab_filename: str) -> "ScriptBPE":
        with PathManager.open(vocab_filename) as vocab_file:
            return cls(cls.load_vocab(vocab_file))

    @staticmethod
    def load_vocab(file: io.IOBase) -> Dict[str, int]:
        def read_words(lines):
            for line in lines:
                if not line.strip():
                    continue
                yield line.strip().split(maxsplit=1)[0]

        words = list(read_words(file))
        num_words = len(words)

        # We don't care about counts, except that we want them to be
        # non-negative and non-overlapping. We want to prioritize pairs
        # which come first in the vocab file. So ignore counts in the file
        # and score them according to reverse of their index in the file.
        return {word: num_words - i for i, word in enumerate(words)}

    @torch.jit.script_method
    def bpe_token(self, token: str) -> List[str]:
        # If full token is in vocab, we're done.
        full_token = token + self.eow
        # `in` not implemented, this should be read `if full_token in self.vocab`
        if self.vocab.get(full_token) is not None:
            return [full_token]

        # Split word into parts, with the last part having EOW attached.
        # Any part (character or char + EOW) not in the vocab on its own
        # should be removed. EOW should always be attached to the last remaining
        # token.
        parts = utf8_chars(token)

        # parts and parts[-1] + self.eow not in self.vocab
        while len(parts) > 0 and self.vocab.get(parts[-1] + self.eow) is None:
            parts.pop()
        # The word consisted entirely of unknown characters
        if len(parts) == 0:
            return [self.eow]
        parts[-1] += self.eow

        # Remove any other obscure characters not in the vocab.
        # No easy way to iterate backwards or create descending ranges,
        # so using a while loop.
        i = 0
        while i < len(parts):
            # parts[i] not in self.vocab
            if self.vocab.get(parts[i]) is None:
                parts.pop(i)
            else:
                i += 1

        # We compare vocab dict scores to this value, so this is where we assume
        # vocab dict values are non-negative.
        NOT_IN_VOCAB = -1
        # break not implemented
        should_break = False

        # Keep going until no more part pairs are in the vocab.
        # In obscure cases this could also get down to a single token, eg. if
        # we filter out some character and rebuild up to a single token.
        while len(parts) > 1 and not should_break:
            # Create part pairs, join part pair with highest score in vocab.
            # In pure python, this could be implemented as
            # max(range(len(parts) - 1),
            #     key=lambda i: self.vocab.get(parts[i] + parts[i+1], -1)))
            max_pair_index = 0
            max_pair_value = NOT_IN_VOCAB
            # We structure the vocabulary to not have ties, but they can come up anyway,
            # for instance in cases with repeated tokens or when passing in vocabs not
            # created with BPE.load_vocab. In the case of a tie between the value of
            # joined segments, they'll be joined proiritizing the first pair in the
            # token according to byte order, ie. left in LTR and right in RTL languages.
            # For instance, if the vocab contains "aa" but not "aaa", then
            # bpe_tokens("aaa") -> ["aa", "a"]. If the vocab contains "ab" and "bc"
            # mapped to the same priority, but not "abc", then
            # bpe_tokens("abc") -> ["ab", "c"].
            for pair_index in range(len(parts) - 1):
                joined = parts[pair_index] + parts[pair_index + 1]
                pair_value = self.vocab.get(joined, NOT_IN_VOCAB)
                if pair_value > max_pair_value:
                    max_pair_value = pair_value
                    max_pair_index = pair_index

            if max_pair_value == NOT_IN_VOCAB:
                # No pairs found in vocab, we're done!
                should_break = True
            else:
                # break, continue not supported; only run this block if we wouldn't
                # want to break out after the above step

                # Combine parts pair with highest priority in vocab.
                # len(parts) shrinks by 1 each iteration, so we should be bounded
                # as linear in token length.
                # Subscript assignment not implemented.
                p1, p2 = parts[max_pair_index : max_pair_index + 2]
                parts = parts[:max_pair_index] + [p1 + p2] + parts[max_pair_index + 2 :]

        return parts

    @torch.jit.script_method
    def tokenize(self, tokens: List[str]) -> List[str]:
        bpe_tokens = torch.jit.annotate(List[str], [])

        for token in tokens:
            # extend not implemented
            for part in self.bpe_token(token):
                bpe_tokens.append(part)

        return bpe_tokens

    def __getstate__(self):
        """These implement pickling for ScriptBPE modules.

        TorchScript models can't be pickled normally. See
        https://github.com/pytorch/pytorch/issues/15116 for more context; in the
        meantime, for TorchScript modules that might want to be pickled
        (this one is often included in say tensorizer/tokenizer state that we want
        in snapshots) we need to implement a custom getstate and setstate for pickling.
        """
        return {"vocab": self.vocab, "eow": self.eow}

    def __setstate__(self, state):
        ScriptBPE.__init__(self, state["vocab"], state["eow"])
