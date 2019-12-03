#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import re
from typing import List, NamedTuple

from fairseq.data.encoders.gpt2_bpe import get_encoder as create_gpt2_bpe
from fairseq.data.encoders.gpt2_bpe_utils import Encoder as GPT2BPEEncoder
from pytext.config import ConfigBase
from pytext.config.component import Component, ComponentType, create_component
from pytext.torchscript.tokenizer import ScriptDoNothingTokenizer
from pytext.utils.file_io import PathManager
from pytorch_pretrained_bert.tokenization import (
    BasicTokenizer,
    WordpieceTokenizer,
    load_vocab,
)
from sentencepiece import SentencePieceProcessor


class Token(NamedTuple):
    value: str
    start: int
    end: int


class Tokenizer(Component):
    """A simple regex-splitting tokenizer."""

    __COMPONENT_TYPE__ = ComponentType.TOKENIZER
    __EXPANSIBLE__ = True

    class Config(Component.Config):
        #: A regular expression for the tokenizer to split on. Tokens are the segments
        #: between the regular expression matches. The start index is inclusive of the
        #: unmatched region, and the end index is exclusive (matching the first
        #: character of the matched split region).
        split_regex: str = r"\s+"
        #: Whether token values should be lowercased or not.
        lowercase: bool = True

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.split_regex, config.lowercase)

    def __init__(self, split_regex=r"\s+", lowercase=True):
        super().__init__(None)
        self.split_regex = split_regex
        self.lowercase = lowercase

    def tokenize(self, input: str) -> List[Token]:
        tokens = []
        start = 0
        tokenize_input = input.lower() if self.lowercase else input
        for match in re.finditer(self.split_regex, tokenize_input):
            split_start, split_end = match.span()
            tokens.append(Token(tokenize_input[start:split_start], start, split_start))
            start = split_end
        tokens.append(Token(tokenize_input[start : len(input)], start, len(input)))
        return [token for token in tokens if token.value]

    def torchscriptify(self):
        raise NotImplementedError


class DoNothingTokenizer(Tokenizer):
    """
    Tokenizer that takes a list of strings and converts to a list of Tokens.
    Useful in cases where tokenizer is run before-hand
    """

    class Config(Component.Config):
        do_nothing: str = ""

    @classmethod
    def from_config(cls, config: Config):
        return cls()

    def __init__(self):
        super().__init__(None)

    def tokenize(self, input: List[str]) -> List[Token]:
        tokens = [Token(token_text, -1, -1) for token_text in input if token_text]
        return tokens

    def torchscriptify(self):
        return ScriptDoNothingTokenizer()


class BERTInitialTokenizer(Tokenizer):
    """
    Basic initial tokenization for BERT.  This is run prior to word piece, does
    white space tokenization in addition to lower-casing and accent removal
    if specified.
    """

    class Config(Tokenizer.Config):
        """Config for this class."""

    @classmethod
    def from_config(cls, config: Config):
        basic_tokenizer = BasicTokenizer(do_lower_case=config.lowercase)
        return cls(basic_tokenizer)

    def __init__(self, basic_tokenizer) -> None:
        self.tokenizer = basic_tokenizer

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        if self.tokenizer.do_lower_case:
            text = self.tokenizer._run_strip_accents(text.lower())
        tokens = self.tokenizer.tokenize(text)
        end = 0
        result = []
        for token in tokens:
            start = text.find(token, end)
            if start == -1:  # safety check, this should not happen
                start = end
            end = start + len(token)
            result.append(Token(token, start, end))
        return result


class WordPieceTokenizer(Tokenizer):
    """Word piece tokenizer for BERT models."""

    class Config(ConfigBase):
        basic_tokenizer: BERTInitialTokenizer.Config = BERTInitialTokenizer.Config()
        wordpiece_vocab_path: str = "/mnt/vol/nlp_technologies/bert/uncased_L-12_H-768_A-12/vocab.txt"

    def __init__(self, wordpiece_vocab, basic_tokenizer, wordpiece_tokenizer) -> None:
        self.vocab = wordpiece_vocab
        self.basic_tokenizer = basic_tokenizer
        self.wordpiece_tokenizer = wordpiece_tokenizer

    @classmethod
    def from_config(cls, config: Config):
        basic_tokenizer = create_component(
            ComponentType.TOKENIZER, config.basic_tokenizer
        )
        vocab = load_vocab(config.wordpiece_vocab_path)
        wordpiece_tokenizer = WordpieceTokenizer(vocab=vocab)
        return cls(vocab, basic_tokenizer, wordpiece_tokenizer)

    def tokenize(self, input_str: str) -> List[Token]:
        tokens = []
        for token in self.basic_tokenizer.tokenize(input_str):
            start = token.start
            for sub_token in self.wordpiece_tokenizer.tokenize(token.value):
                piece_len = (
                    len(sub_token)
                    if not sub_token.startswith("##")
                    else (len(sub_token) - 2)  # account for ##
                )
                if sub_token == "[UNK]":
                    # this fixes the bug wherein piece_len = 5 for all [UNK]
                    piece_len = len(token.value)
                end = start + piece_len
                tokens.append(Token(sub_token, start, end))
                start = end
        return [token for token in tokens if token.value]


class PickleableGPT2BPEEncoder(GPT2BPEEncoder):
    """Fairseq's encoder stores the regex module as a local reference on its encoders,
    which means they can't be saved via pickle.dumps or torch.save. This modified
    their save/load logic doesn't store the module, and restores the reference
    after re-inflating."""

    def __getstate__(self):
        # make a shallow copy of state to avoid side effect on the original object
        state = copy.copy(vars(self))
        state.pop("re")
        return state

    def __setstate__(self, state):
        vars(self).update(state)
        import regex

        self.re = regex


class GPT2BPETokenizer(Tokenizer):
    """Tokenizer for gpt-2 and RoBERTa."""

    class Config(ConfigBase):
        bpe_encoder_path: str = (
            "manifold://pytext_training/tree/static/vocabs/bpe/gpt2/encoder.json"
        )
        bpe_vocab_path: str = (
            "manifold://pytext_training/tree/static/vocabs/bpe/gpt2/vocab.bpe"
        )

    @classmethod
    def from_config(cls, config: Config):
        # TODO: T57433776 remove once FairSeq support PathManager
        config.bpe_encoder_path = PathManager.get_local_path(config.bpe_encoder_path)
        config.bpe_vocab_path = PathManager.get_local_path(config.bpe_vocab_path)

        bpe = create_gpt2_bpe(config.bpe_encoder_path, config.bpe_vocab_path)
        # This hacks the bpe instance to be picklable
        bpe = copy.copy(bpe)
        bpe.__class__ = PickleableGPT2BPEEncoder

        return cls(bpe)

    def __init__(self, bpe: GPT2BPEEncoder):
        self.bpe = bpe

    def tokenize(self, input_str: str) -> List[Token]:
        bpe_ids = self.bpe.encode(input_str)
        char_tokens = [self.bpe.decoder[id].lstrip(u"\u0120") for id in bpe_ids]
        # fix for incorrect decoding of utf-8 chars
        for i, char_token in enumerate(char_tokens):
            try:
                char_tokens[i] = bytearray(
                    [self.bpe.byte_decoder[char] for char in char_token]
                ).decode("utf-8")
            # handles BPE breaking a single multi-byte char into pieces
            except UnicodeDecodeError:
                continue
        lengths = [len(token) for token in char_tokens]
        tokens = []
        end = 0
        for length, id, char_token in zip(lengths, bpe_ids, char_tokens):
            start = input_str.find(char_token, end)
            end = start + length
            tokens.append(Token(str(id), start, end))
            # handles bad start/end indices cascading to subsequent tokens.
            if len(tokens) > 1 and end < tokens[-2].end:
                end = tokens[-2].end
        return [token for token in tokens if token.value]


class CppProcessorMixin:
    """Cpp processors like SentencePiece don't pickle well; reload them."""

    def _load_processor(self):
        raise NotImplementedError

    def __getstate__(self):
        state = dict(vars(self))
        state.pop("processor")
        return state

    def __setstate__(self, state):
        vars(self).update(state)
        self._load_processor()


class SentencePieceTokenizer(Tokenizer, CppProcessorMixin):
    """Sentence piece tokenizer."""

    class Config(ConfigBase):
        sp_model_path: str = ""

    def __init__(self, sp_model_path: str = ""):
        self.sp_model_path = sp_model_path
        self._load_processor()

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.sp_model_path)

    def tokenize(self, input_str: str) -> List[Token]:
        pieces = self.processor.EncodeAsPieces(input_str)
        return [Token(piece, -1, -1) for piece in pieces]

    def _load_processor(self):
        self.processor = SentencePieceProcessor()
        self.processor.Load(self.sp_model_path)
