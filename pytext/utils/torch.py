#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import io
from typing import Dict, List, Optional, Tuple

import torch


# ===== the following section should be replaced once JIT provide native support
@torch.jit.script
def list_max(l: List[int]):
    max_value = l[0]  # fine to throw if empty
    for i in range(len(l) - 1):  # don't forget the +1
        max_value = max(max_value, l[i + 1])
    return max_value


@torch.jit.script
def list_membership(item: int, list: List[int]):
    item_present = False
    for i in list:
        if item == i:
            item_present = True
    return item_present


@torch.jit.script
def reverse_tensor_list(int_list: List[torch.Tensor]) -> List[torch.Tensor]:
    l_len = len(int_list)
    res = []
    for idx in range(l_len):
        res.append(int_list[l_len - idx - 1])
    return res


@torch.jit.script
def long_tensor_2d(shape: Tuple[int, int], fill_value: int = 0) -> torch.Tensor:
    """Return a new 2d torch.LongTensor with size according to shape.
    The values of this tensor will be fill_value."""
    outer = torch.jit.annotate(List[List[int]], [])
    inner = torch.jit.annotate(List[int], [])
    for _i in range(shape[1]):
        inner.append(fill_value)
    for _i in range(shape[0]):
        outer.append(inner)
    return torch.tensor(outer, dtype=torch.long)


@torch.jit.script
def pad_2d_mask(
    input: List[List[int]], pad_value: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list to a 2d tensor. Returns a pair of tensors, the padded tensor
    as well as a mask tensor. The mask tensor has the same shape as the padded tensor,
    with a 1 in the position of non-pad values and a 0 in the position of pads."""
    max_len = 0
    for i in input:
        max_len = max(max_len, len(i))
    tensor = long_tensor_2d((len(input), max_len), pad_value)
    mask = long_tensor_2d((len(input), max_len), 0)
    for i in range(len(input)):
        for j in range(len(input[i])):
            tensor[i][j] = input[i][j]
            mask[i][j] = 1
    return tensor, mask


# ========= end section


class Vocabulary(torch.jit.ScriptModule):
    def __init__(
        self,
        vocab_list,
        unk_idx: int = 0,
        pad_idx: int = -1,
        bos_idx: int = -1,
        eos_idx: int = -1,
    ):
        super().__init__()
        self.vocab = torch.jit.Attribute(vocab_list, List[str])
        self.unk_idx = torch.jit.Attribute(unk_idx, int)
        self.pad_idx = torch.jit.Attribute(pad_idx, int)
        self.eos_idx = torch.jit.Attribute(eos_idx, int)
        self.bos_idx = torch.jit.Attribute(bos_idx, int)
        self.idx = torch.jit.Attribute(
            {word: i for i, word in enumerate(vocab_list)}, Dict[str, int]
        )

    @torch.jit.script_method
    def lookup_indices_1d(self, values: List[str]) -> List[int]:
        result = torch.jit.annotate(List[int], [])
        for value in values:
            result.append(self.idx.get(value, self.unk_idx))
        return result

    @torch.jit.script_method
    def lookup_indices_2d(self, values: List[List[str]]) -> List[List[int]]:
        result = torch.jit.annotate(List[List[int]], [])
        for value in values:
            result.append(self.lookup_indices_1d(value))
        return result

    @torch.jit.script_method
    def lookup_words_1d(
        self,
        values: torch.Tensor,
        filter_token_list: List[int] = (),
        possible_unk_token: Optional[str] = None,
    ) -> List[str]:
        """If possible_unk_token is not None, then all UNK id's will be replaced
        by possible_unk_token instead of the default UNK string which is <UNK>.
        This is a simple way to resolve UNK's when there's a correspondence
        between source and target translations.
        """
        result = torch.jit.annotate(List[str], [])
        for idx in range(values.size(0)):
            value = int(values[idx])
            if not list_membership(value, filter_token_list):
                result.append(self.lookup_word(value, possible_unk_token))
        return result

    @torch.jit.script_method
    def lookup_word(self, idx: int, possible_unk_token: Optional[str] = None):
        if idx < len(self.vocab):
            return self.vocab[idx]
        else:
            return (
                self.vocab[self.unk_idx]
                if possible_unk_token is None
                else possible_unk_token
            )


@torch.jit.script
def utf8_chars(s: str) -> List[str]:
    """An implementation of UTF8 character iteration in TorchScript.
    There are no bitwise operations in torchscript, so we compare directly to
    integer values. There isn't a lot of validation, for instance if you pass
    in an improperly encoded string with an out-of-place continuation byte,
    or with a non-left-to-right byte order, you'll get unexpected results
    and likely throw. Torch itself takes in unicode strings and encodes them
    as UTF8, so that should be actively hard to do.

    The logic is simple: looking at the current start-of-character byte.
    If its high bit is 0, it's a 1-byte character. Otherwise, the number of
    bytes is the number of leading 1s in its binary representation, so
    find that number by comparing it directly to ints with the appropriate
    representation, then append that many bytes as a character and move past
    them to the next start byte.
    """
    chars = torch.jit.annotate(List[str], [])
    i = 0
    while i < len(s):
        byte = ord(s[i])
        if byte < 0b10000000:
            chars.append(s[i])
            i += 1
        else:
            if byte < 0b11100000:
                num_bytes = 2
            elif byte < 0b11110000:
                num_bytes = 3
            elif byte < 0b11111000:
                num_bytes = 4
            elif byte < 0b11111100:
                num_bytes = 5
            elif byte < 0b11111110:
                num_bytes = 6
            elif byte < 0b11111111:
                num_bytes = 7
            else:
                num_bytes = 8
            chars.append(s[i : i + num_bytes])
            i += num_bytes
    return chars


class BPE(torch.jit.ScriptModule):
    """Byte-pair encoding implementation in TorchScript.

    vocab_file should be a file-like object separated by newlines, where each line
    consists of a word and a count separated by whitespace. Words in the vocab
    therefore can't contain space (according to python regex \\s). The vocab file
    should be sorted according to the importance of each token, and they will be
    merged in this priority; the actual score values are irrelevant.

    eow_token should be a string that is appended to the last character and token,
    and that token is used at each step in the process and returned at the end.
    You should set this to be consistent with the EOW signature used however you
    generated your BPE vocab file.

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
    >>> bpe = BPE.from_vocab_file(vocab_file)
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
    def from_vocab_file(cls, vocab_file: io.IOBase) -> "BPE":
        return cls(cls.load_vocab(vocab_file))

    @classmethod
    def from_vocab_filename(cls, vocab_filename: str) -> "BPE":
        with open(vocab_filename) as vocab_file:
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
            return parts
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


@torch.jit.script
def make_sequence_lengths(batch: List[List[str]]) -> List[int]:
    seq_lens = torch.jit.annotate(List[int], [])
    for sentence in batch:
        seq_lens.append(len(sentence))
    return seq_lens


@torch.jit.script
def pad_2d(
    batch: List[List[int]], seq_lens: List[int], pad_idx: int
) -> List[List[int]]:
    pad_to_length = list_max(seq_lens)
    for sentence in batch:
        for _ in range(pad_to_length - len(sentence)):
            sentence.append(pad_idx)
    return batch


@torch.jit.script
def add_bos_eos_2d(
    values: List[List[int]], special_token: int = 0, use_first_value: bool = False
) -> List[List[int]]:
    results = torch.jit.annotate(List[List[int]], [])
    for value in values:
        result = torch.jit.annotate(List[int], [])
        if use_first_value and len(value) > 0:
            special_token = value[0]
        result.append(special_token)
        result.extend(value)
        result.append(special_token)
        results.append(result)
    return results


@torch.jit.script
def make_byte_inputs(
    batch: List[List[str]], max_byte_len: int, offset_for_non_padding: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_lens = make_sequence_lengths(batch)
    max_num_tokens = list_max(seq_lens)
    bytes = torch.zeros(len(batch), max_num_tokens, max_byte_len, dtype=torch.long)

    for batch_index in range(len(batch)):
        sentence = batch[batch_index]
        for token_index in range(len(sentence)):
            token = sentence[token_index]
            for byte_index in range(min(len(token), max_byte_len)):
                s = token[byte_index]
                # use empty string as eos because torchscript not support chr()
                if s == "":
                    v = 256
                else:
                    v = ord(s)
                # add offset_for_non_padding to conform to Fairseq pre-training
                bytes[batch_index][token_index][byte_index] = v + offset_for_non_padding

    return bytes, torch.tensor(seq_lens)
