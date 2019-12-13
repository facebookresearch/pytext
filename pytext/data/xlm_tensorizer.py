#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
from typing import Any, Dict, List, Optional, Tuple

import torch
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from pytext.config.component import ComponentType, create_component
from pytext.data.bert_tensorizer import (
    BERTTensorizerBase,
    BERTTensorizerBaseScriptImpl,
    build_fairseq_vocab,
)
from pytext.data.tensorizers import lookup_tokens
from pytext.data.tokenizers import Tokenizer
from pytext.data.utils import EOS, MASK, PAD, UNK, Vocabulary
from pytext.data.xlm_constants import LANG2ID_15
from pytext.torchscript.tensorizer import ScriptXLMTensorizer
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils.file_io import PathManager


class XLMTensorizerScriptImpl(BERTTensorizerBaseScriptImpl):
    def __init__(
        self,
        tokenizer: Tokenizer,
        vocab: Vocabulary,
        max_seq_len: int,
        language_vocab: List[str],
        default_language: str,
    ):
        super().__init__(tokenizer, vocab, max_seq_len)
        self.language_vocab = ScriptVocabulary(language_vocab)
        self.default_language = torch.jit.Attribute(default_language, str)

    def _lookup_tokens(
        self, tokens: List[Tuple[str, int, int]], max_seq_len: Optional[int] = None
    ) -> Tuple[List[int], List[int], List[int]]:
        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        return self.vocab_lookup(
            tokens,
            bos_idx=self.vocab.eos_idx,
            eos_idx=self.vocab.eos_idx,
            use_eos_token_for_bos=True,
            max_seq_len=max_seq_len,
        )

    def numberize(
        self,
        per_sentence_tokens: List[List[Tuple[str, int, int]]],
        per_sentence_languages: List[int],
    ) -> Tuple[List[int], List[int], int, List[int]]:
        tokens: List[int] = []
        segment_labels: List[int] = []  # e.g language_ids
        seq_len: int = 0
        positions: List[int] = []
        max_seq_len: int = self.max_seq_len // len(per_sentence_tokens)

        for idx, single_sentence_tokens in enumerate(per_sentence_tokens):
            lookup_ids: List[int] = self._lookup_tokens(
                single_sentence_tokens, max_seq_len=max_seq_len
            )[0]
            lookup_ids = self._wrap_numberized_tokens(lookup_ids, idx)

            tokens.extend(lookup_ids)
            segment_labels.extend([per_sentence_languages[idx]] * len(lookup_ids))
        seq_len = len(tokens)
        positions = [i for i in range(seq_len)]

        return tokens, segment_labels, seq_len, positions

    def forward(
        self,
        texts: Optional[List[List[str]]] = None,
        pre_tokenized: Optional[List[List[List[str]]]] = None,
        languages: Optional[List[List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Wire up tokenize(), numberize() and tensorize() functions for data
        processing.
        """
        batch_size: int = self.batch_size(texts, pre_tokenized)
        row_size: int = self.row_size(texts, pre_tokenized)
        if languages is None:
            languages = [[self.default_language] * row_size] * batch_size

        tokens_2d: List[List[int]] = []
        segment_labels_2d: List[List[int]] = []
        seq_lens_1d: List[int] = []
        positions_2d: List[List[int]] = []

        for idx in range(self.batch_size(texts, pre_tokenized)):
            tokens: List[List[Tuple[str, int, int]]] = self.tokenize(
                self.get_texts_by_index(texts, idx),
                self.get_tokens_by_index(pre_tokenized, idx),
            )
            language_ids: List[int] = [
                self.language_vocab.idx.get(
                    languages[idx][0], self.language_vocab.unk_idx
                )
            ] * row_size

            numberized: Tuple[List[int], List[int], int, List[int]] = self.numberize(
                tokens, language_ids
            )
            tokens_2d.append(numberized[0])
            segment_labels_2d.append(numberized[1])
            seq_lens_1d.append(numberized[2])
            positions_2d.append(numberized[3])

        return self.tensorize(tokens_2d, segment_labels_2d, seq_lens_1d, positions_2d)


class XLMTensorizer(BERTTensorizerBase):
    """
    Tensorizer for Cross-lingual LM tasks. Works for single sentence as well
    as sentence pair.
    """

    __TENSORIZER_SCRIPT_IMPL__ = XLMTensorizerScriptImpl

    class Config(BERTTensorizerBase.Config):
        vocab_file: str = "/mnt/vol/nlp_technologies/xlm/vocab_xnli_15"
        tokenizer: Tokenizer.Config = Tokenizer.Config()
        max_vocab: int = 95000
        min_count: int = 0
        # language identifiers for extracting the language from a row of data
        # during numberize
        language_column: str = "language"
        # language-to-id mapping used to obtain language embeddings
        lang2id: Dict[str, int] = LANG2ID_15
        # Controls whether language is being read from the data file (which
        # is what happens for finetuning) or being added during processing
        # (which is what happens during pretraining)
        has_language_in_data: bool = False
        # controls whether we train with language embeddings or not
        use_language_embeddings: bool = True

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        with PathManager.open(config.vocab_file) as file_path:
            vocab = build_fairseq_vocab(
                dictionary_class=MaskedLMDictionary,
                vocab_file=file_path,
                max_vocab=config.max_vocab,
                min_count=config.min_count,
                special_token_replacements={
                    "<unk>": UNK,
                    "<pad>": PAD,
                    "</s>": EOS,
                    "<mask>": MASK,
                },
            )
        return cls(
            columns=config.columns,
            vocab=vocab,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            language_column=config.language_column,
            lang2id=config.lang2id,
            use_language_embeddings=config.use_language_embeddings,
            has_language_in_data=config.has_language_in_data,
        )

    def __init__(
        self,
        columns: List[str] = Config.columns,
        vocab: Vocabulary = None,
        tokenizer: Tokenizer = None,
        max_seq_len: int = Config.max_seq_len,
        language_column: str = Config.language_column,
        lang2id: Dict[str, int] = Config.lang2id,
        use_language_embeddings: bool = Config.use_language_embeddings,
        has_language_in_data: bool = Config.has_language_in_data,
    ) -> None:
        assert len(columns) <= 2, "More than 2 text fields are not supported."

        super().__init__(
            columns=columns, vocab=vocab, tokenizer=tokenizer, max_seq_len=max_seq_len
        )
        self.language_column = language_column
        self.lang2id = lang2id
        self.use_language_embeddings = use_language_embeddings
        self.has_language_in_data = has_language_in_data
        # unlike BERT, XLM uses the EOS token for both beginning and end of
        # sentence
        self.bos_token = self.vocab.eos_token
        self.default_language = "en"

    @property
    def column_schema(self):
        schema = super().column_schema
        if self.has_language_in_data:
            schema += [(self.language_column, str)]
        return schema

    def get_lang_id(self, row: Dict, col: str) -> int:
        # generate lang embeddings. if training without lang embeddings, use
        # the first language as the lang_id (there will always be one lang)
        if self.use_language_embeddings:
            lang = row[col]
            assert lang in self.lang2id, f"language {lang} not supported in {row}"
            lang_id = self.lang2id[lang]
            return lang_id
        else:
            # use En as default
            return self.lang2id.get(self.default_language, 0)

    def _lookup_tokens(self, text: str, seq_len: int) -> List[str]:
        return lookup_tokens(
            text,
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            bos_token=self.vocab.eos_token,
            eos_token=self.vocab.eos_token,
            use_eos_token_for_bos=True,
            max_seq_len=seq_len,
        )

    def numberize(self, row: Dict) -> Tuple[Any, ...]:
        sentences = []
        language_column = self.language_column
        columns = self.columns

        # sequence_length is adjusted based on the number of text fields and needs
        # to account for the special tokens which we will be wrapping
        seq_len = self.max_seq_len // len(columns)
        sentences = [
            self._lookup_tokens(row[column], seq_len)[0] for column in self.columns
        ]
        seq_lens = [len(sentence) for sentence in sentences]
        lang_ids = [self.get_lang_id(row, language_column)] * len(self.columns)
        # expand the language ids to each token
        lang_ids = ([lang_id] * seq_len for lang_id, seq_len in zip(lang_ids, seq_lens))

        tokens = list(itertools.chain(*sentences))
        segment_labels = list(itertools.chain(*lang_ids))
        seq_len = len(tokens)
        positions = [index for index in range(seq_len)]
        return tokens, segment_labels, seq_len, positions

    def torchscriptify(self):
        languages = [0] * (max(list(self.lang2id.values())) + 1)
        for k, v in self.lang2id.items():
            languages[v] = k

        return ScriptXLMTensorizer(
            tokenizer=self.tokenizer.torchscriptify(),
            token_vocab=ScriptVocabulary(
                list(self.vocab),
                pad_idx=self.vocab.get_pad_index(),
                bos_idx=self.vocab.get_eos_index(),
                eos_idx=self.vocab.get_eos_index(),
                unk_idx=self.vocab.get_unk_index(),
            ),
            language_vocab=ScriptVocabulary(languages),
            max_seq_len=self.max_seq_len,
            default_language=self.default_language,
        )
