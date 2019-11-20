#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
from typing import Any, Dict, List, Tuple

from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from pytext.config.component import ComponentType, create_component
from pytext.data.bert_tensorizer import BERTTensorizerBase, build_fairseq_vocab
from pytext.data.tensorizers import lookup_tokens
from pytext.data.tokenizers import Tokenizer
from pytext.data.utils import EOS, MASK, PAD, UNK, Vocabulary
from pytext.data.xlm_constants import LANG2ID_15
from pytext.torchscript.tensorizer import ScriptXLMTensorizer
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils.file_io import PathManager


class XLMTensorizer(BERTTensorizerBase):
    """
    Tensorizer for Cross-lingual LM tasks. Works for single sentence as well
    as sentence pair.
    """

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
