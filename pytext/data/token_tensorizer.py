#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional, Tuple

import torch
from pytext.config.component import ComponentType, create_component
from pytext.data.tensorizers import TensorizerScriptImpl
from pytext.data.tokenizers import Tokenizer
from pytext.torchscript.tensorizer.tensorizer import VocabLookup
from pytext.torchscript.tokenizer import ScriptDoNothingTokenizer
from pytext.torchscript.utils import ScriptBatchInput, pad_2d
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils import cuda
from pytext.utils.file_io import PathManager
from pytext.utils.lazy import lazy_property

from .tensorizers import Tensorizer, VocabConfig, tokenize
from .utils import VocabBuilder, Vocabulary


class TokenTensorizerScriptImpl(TensorizerScriptImpl):
    def __init__(
        self,
        add_bos_token: bool,
        add_eos_token: bool,
        use_eos_token_for_bos: bool,
        max_seq_len: int,
        vocab: Vocabulary,
        tokenizer: Optional[Tokenizer],
    ):
        super().__init__()

        if tokenizer is not None and hasattr(tokenizer, "torchscriptify"):
            try:
                self.tokenizer = tokenizer.torchscriptify()
            except NotImplementedError:
                # This is fine as long as the exported tokenizer is only used
                # in pre-tokenized mode
                self.tokenizer = None
        else:
            self.tokenizer = None

        self.do_nothing_tokenizer = ScriptDoNothingTokenizer()
        self.vocab = ScriptVocabulary(
            list(vocab),
            pad_idx=vocab.get_pad_index(),
            bos_idx=vocab.get_bos_index() if add_bos_token else -1,
            eos_idx=vocab.get_eos_index() if add_eos_token else -1,
        )
        self.vocab_lookup_1d = VocabLookup(self.vocab)

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_eos_token_for_bos = use_eos_token_for_bos
        self.max_seq_len = max_seq_len

    def get_texts_by_index(
        self, texts: Optional[List[List[str]]], index: int
    ) -> Optional[str]:
        if texts is None or len(texts) == 0:
            return None

        # TokenTensorizer only works with a single text per row, stick with that
        return texts[index][0]

    def get_tokens_by_index(
        self, tokens: Optional[List[List[List[str]]]], index: int
    ) -> Optional[List[str]]:
        if tokens is None or len(tokens) == 0:
            return None

        # TokenTensorizer only works with a single text per row, stick with that
        return tokens[index][0]

    def _lookup_tokens_1d(
        self, tokens: List[Tuple[str, int, int]]
    ) -> Tuple[List[int], List[int], List[int]]:
        return self.vocab_lookup_1d(
            tokens,
            bos_idx=self.vocab.bos_idx if self.add_bos_token else None,
            eos_idx=self.vocab.eos_idx if self.add_eos_token else None,
            use_eos_token_for_bos=self.use_eos_token_for_bos,
            max_seq_len=self.max_seq_len,
        )

    def tokenize(
        self, row_text: Optional[str], row_pre_tokenized: Optional[List[str]]
    ) -> List[Tuple[str, int, int]]:

        tokens: List[Tuple[str, int, int]] = []
        if row_text is not None:
            if self.tokenizer is not None:
                tokens = self.tokenizer.tokenize(row_text)
        elif row_pre_tokenized is not None:
            for token in row_pre_tokenized:
                tokens.extend(self.do_nothing_tokenizer.tokenize(token))

        return tokens

    def numberize(
        self, text_tokens: List[Tuple[str, int, int]]
    ) -> Tuple[List[int], int, List[Tuple[int, int]]]:
        token_indices: List[int] = []
        token_starts: List[int] = []
        token_ends: List[int] = []

        token_indices, token_starts, token_ends = self._lookup_tokens_1d(text_tokens)

        token_ranges: List[Tuple[int, int]] = []

        for s, e in zip(token_starts, token_ends):
            token_ranges.append((s, e))

        return token_indices, len(token_indices), token_ranges

    def tensorize(
        self,
        tokens_2d: List[List[int]],
        seq_lens_1d: List[int],
        positions_2d: List[List[Tuple[int, int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        token_indices_tensor: torch.Tensor = torch.tensor(
            pad_2d(tokens_2d, seq_lens=seq_lens_1d, pad_idx=self.vocab.pad_idx),
            dtype=torch.long,
        )

        token_starts_2d: List[List[int]] = []
        token_ends_2d: List[List[int]] = []

        for position_list in positions_2d:
            token_starts_2d.append([x[0] for x in position_list])
            token_ends_2d.append([x[1] for x in position_list])

        token_positions_tensor = torch.stack(
            [
                torch.tensor(
                    pad_2d(token_starts_2d, seq_lens=seq_lens_1d, pad_idx=-1),
                    dtype=torch.long,
                ),
                torch.tensor(
                    pad_2d(token_ends_2d, seq_lens=seq_lens_1d, pad_idx=-1),
                    dtype=torch.long,
                ),
            ],
            dim=2,
        )

        return (
            token_indices_tensor,
            torch.tensor(seq_lens_1d, dtype=torch.long),
            token_positions_tensor,
        )

    def forward(
        self, inputs: ScriptBatchInput
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        tokens_2d: List[List[int]] = []
        seq_lens_1d: List[int] = []
        positions_2d: List[List[Tuple[int, int]]] = []

        for idx in range(self.batch_size(inputs)):
            tokens: List[Tuple[str, int, int]] = self.tokenize(
                self.get_texts_by_index(inputs.texts, idx),
                self.get_tokens_by_index(inputs.tokens, idx),
            )

            numberized: Tuple[List[int], int, List[Tuple[int, int]]] = self.numberize(
                tokens
            )
            tokens_2d.append(numberized[0])
            seq_lens_1d.append(numberized[1])
            positions_2d.append(numberized[2])

        return self.tensorize(tokens_2d, seq_lens_1d, positions_2d)


class ScriptBasedTokenTensorizer(Tensorizer):
    """
    An Implementation of TokenTensorizer that uses a TorchScript module in the
    background and is hence torchscriptifiable.

    Note that unlike the original TokenTensorizer, this version cannot deal with
    arbitrarily nested lists of tokens.
    """

    __TENSORIZER_SCRIPT_IMPL__ = TokenTensorizerScriptImpl

    class Config(Tensorizer.Config):
        #: The name of the text column to parse from the data source.
        column: str = "text"
        #: The tokenizer to use to split input text into tokens.
        tokenizer: Tokenizer.Config = Tokenizer.Config()
        add_bos_token: bool = False
        add_eos_token: bool = False
        use_eos_token_for_bos: bool = False
        max_seq_len: Optional[int] = None
        vocab: VocabConfig = VocabConfig()
        vocab_file_delimiter: str = " "

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        return cls(
            text_column=config.column,
            tokenizer=tokenizer,
            add_bos_token=config.add_bos_token,
            add_eos_token=config.add_eos_token,
            use_eos_token_for_bos=config.use_eos_token_for_bos,
            max_seq_len=config.max_seq_len,
            vocab_config=config.vocab,
            vocab_file_delimiter=config.vocab_file_delimiter,
            is_input=config.is_input,
        )

    def __init__(
        self,
        text_column,
        tokenizer=None,
        add_bos_token=Config.add_bos_token,
        add_eos_token=Config.add_eos_token,
        use_eos_token_for_bos=Config.use_eos_token_for_bos,
        max_seq_len=Config.max_seq_len,
        vocab_config=None,
        vocab=None,
        vocab_file_delimiter=" ",
        is_input=Config.is_input,
    ):
        self.text_column = text_column
        self.tokenizer = tokenizer or Tokenizer()
        self.vocab = vocab
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_eos_token_for_bos = use_eos_token_for_bos
        self.max_seq_len = max_seq_len or 2 ** 30  # large number
        self.vocab_builder = None
        self.vocab_config = vocab_config or VocabConfig()
        self.vocab_file_delimiter = vocab_file_delimiter
        super().__init__(is_input)

    @property
    def column_schema(self):
        return [(self.text_column, str)]

    def _reverse_lookup(self, token_ids):
        return [self.vocab[id] for id in token_ids]

    def initialize(self, vocab_builder=None, from_scratch=True):
        """Build vocabulary based on training corpus."""
        if self.vocab and from_scratch:
            if self.vocab_config.build_from_data or self.vocab_config.vocab_files:
                print(
                    f"`{self.text_column}` column: vocab already provided, skipping "
                    f"adding tokens from data and from vocab files."
                )
            return

        if not self.vocab_config.build_from_data and not self.vocab_config.vocab_files:
            raise ValueError(
                f"To create token tensorizer for '{self.text_column}', either "
                f"`build_from_data` or `vocab_files` must be set."
            )
        if not self.vocab_builder:
            # else means not initialize from scratch, self.vocab_builder
            # would be set already
            self.vocab_builder = vocab_builder or VocabBuilder(
                delimiter=self.vocab_file_delimiter
            )
            self.vocab_builder.use_bos = self.add_bos_token
            self.vocab_builder.use_eos = self.add_eos_token
        if not self.vocab_config.build_from_data:
            self._add_vocab_from_files()
            self.vocab = self.vocab_builder.make_vocab()
            return

        try:
            while True:
                row = yield
                raw_text = row[self.text_column]
                tokenized = self.tokenizer.tokenize(raw_text)
                self.vocab_builder.add_all([t.value for t in tokenized])
        except GeneratorExit:
            self.vocab_builder.truncate_to_vocab_size(
                self.vocab_config.size_from_data, self.vocab_config.min_counts
            )
            self._add_vocab_from_files()
            self.vocab = self.vocab_builder.make_vocab()

    def _add_vocab_from_files(self):
        for vocab_file in self.vocab_config.vocab_files:
            with PathManager.open(vocab_file.filepath) as f:
                self.vocab_builder.add_from_file(
                    f,
                    vocab_file.skip_header_line,
                    vocab_file.lowercase_tokens,
                    vocab_file.size_limit,
                )

    def _tokenize(self, text=None, pre_tokenized=None, add_eos_bos=True):

        add_bos = self.add_bos_token and add_eos_bos
        add_eos = self.add_eos_token and add_eos_bos

        return tokenize(
            text=text,
            pre_tokenized=pre_tokenized,
            tokenizer=self.tokenizer,
            bos_token=self.vocab.bos_token if add_bos else None,
            eos_token=self.vocab.eos_token if add_eos else None,
            pad_token=self.vocab.pad_token,
            use_eos_token_for_bos=self.use_eos_token_for_bos,
            max_seq_len=self.max_seq_len,
        )

    @lazy_property
    def tensorizer_script_impl(self):
        return self.__TENSORIZER_SCRIPT_IMPL__(
            add_bos_token=self.add_bos_token,
            add_eos_token=self.add_eos_token,
            use_eos_token_for_bos=self.use_eos_token_for_bos,
            max_seq_len=self.max_seq_len,
            vocab=self.vocab,
            tokenizer=self.tokenizer,
        )

    def numberize(self, row):
        """
        Tokenize and look up in vocabulary.

        A few notable things:

        1) We're using the non-torchsciptified tokenizer here.
        This allows us to use non-torchscriptifiable tokenizers if we don't intend
        to torchscriptify this module.

        2) When using the ScriptImpl to do the lookup, it takes care of the
        BOS / EOS stuff there. Hence we don't need to do that with the tokenizer.

        3) The tokenize function from tensorizer.py returns a tuple of
        (tokens, start_indices, end_indices), while the ScriptImpl expects a
        list of (token, start_idx, end_idx) tuples so we need to unzip these

        """
        return self.tensorizer_script_impl.numberize(
            list(zip(*self._tokenize(text=row[self.text_column], add_eos_bos=False)))
        )

    def prepare_input(self, row):
        """
        Tokenize, look up in vocabulary, return tokenized_texts in raw text

        Similarly to the above function, tokenization is done with the original
        and not the torchscriptified tokenizer.
        """
        tokenized_texts, start_idx, end_idx = self._tokenize(row[self.text_column])
        token_ranges = list(zip(start_idx, end_idx))
        return list(tokenized_texts), len(tokenized_texts), token_ranges

    def tensorize(self, batch):
        (
            token_indices_tensor,
            seq_lens_1d,
            token_positions_tensor,
        ) = self.tensorizer_script_impl.tensorize_wrapper(*zip(*batch))

        # Need to map them to cuda tensors so that we can run this on GPU
        return (
            cuda.tensor(token_indices_tensor, dtype=torch.long),
            cuda.tensor(seq_lens_1d, dtype=torch.long),
            cuda.tensor(token_positions_tensor, dtype=torch.long),
        )

    def sort_key(self, row):
        # use seq_len as sort key
        return row[1]
