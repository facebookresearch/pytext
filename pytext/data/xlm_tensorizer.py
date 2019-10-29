#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
from typing import Any, Dict, List, Optional, Tuple

from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from pytext.config.component import ComponentType, create_component
from pytext.data.bert_tensorizer import BERTTensorizer, build_fairseq_vocab
from pytext.data.tensorizers import lookup_tokens
from pytext.data.tokenizers import Tokenizer
from pytext.data.utils import EOS, MASK, PAD, UNK, Vocabulary
from pytext.data.xlm_constants import LANG2ID_15


class XLMTensorizer(BERTTensorizer):
    """
    Tensorizer for Cross-lingual LM tasks. Works for single sentence as well
    as sentence pair (as long as the pair have the same language).
    """

    class Config(BERTTensorizer.Config):
        columns: List[str] = ["text"]
        # Parameters related to the vocab
        vocab_file: str = "/mnt/vol/nlp_technologies/xlm/vocab_xnli_15"
        max_vocab: int = 95000
        min_count: int = 0
        # Special tokens need to be specified in order to update PyText defaults
        eos_token: str = "</s>"
        pad_token: str = "<pad>"
        unk_token: str = "<unk>"
        mask_token: str = "<mask>"

        # Parameters related to the tokenizer and numberization
        tokenizer: Tokenizer.Config = Tokenizer.Config()
        add_bos_token: bool = True
        add_eos_token: bool = True
        use_eos_token_for_bos: bool = True
        # See parent class for more details about this flag
        wrap_special_tokens: bool = False

        max_seq_len: Optional[int] = 256
        # controls whether we train with language embeddings or not
        use_language_embeddings: bool = True
        # language identifiers for extracting the language from a row of data
        # during numberize
        language_columns: List[str] = ["language"]
        # language-to-id mapping used to obtain language embeddings
        lang2id: Dict[str, int] = LANG2ID_15
        # Controls whether language is being read from the data file (which
        # is what happens for finetuning) or being added during processing
        # (which is what happens during pretraining)
        has_language_in_data: bool = False

        # Deprecated parameters
        # TODO: Remove these after adding the necessary config adapters
        is_fairseq: bool = False
        pretraining: bool = False
        reset_positions: bool = False

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        special_token_replacements = {
            config.unk_token: UNK,
            config.pad_token: PAD,
            config.eos_token: EOS,
            config.mask_token: MASK,
        }
        vocab = build_fairseq_vocab(
            dictionary_class=MaskedLMDictionary,
            vocab_file=config.vocab_file,
            special_token_replacements=special_token_replacements,
            max_vocab=config.max_vocab,
            min_count=config.min_count,
        )
        return cls(
            columns=config.columns,
            vocab=vocab,
            tokenizer=tokenizer,
            add_bos_token=config.add_bos_token,
            add_eos_token=config.add_eos_token,
            use_eos_token_for_bos=config.use_eos_token_for_bos,
            wrap_special_tokens=config.wrap_special_tokens,
            max_seq_len=config.max_seq_len,
            language_columns=config.language_columns,
            lang2id=config.lang2id,
            use_language_embeddings=config.use_language_embeddings,
            has_language_in_data=config.has_language_in_data,
        )

    def __init__(
        self,
        columns: List[str],
        vocab: Vocabulary,
        tokenizer: Tokenizer,
        add_bos_token: bool = Config.add_bos_token,
        add_eos_token: bool = Config.add_eos_token,
        use_eos_token_for_bos: bool = Config.use_eos_token_for_bos,
        wrap_special_tokens: bool = Config.wrap_special_tokens,
        max_seq_len: int = Config.max_seq_len,
        language_columns=Config.language_columns,
        lang2id=Config.lang2id,
        use_language_embeddings=Config.use_language_embeddings,
        has_language_in_data=Config.has_language_in_data,
    ) -> None:
        assert len(columns) <= 2, "More than 2 text fields are not supported."

        assert len(language_columns) == 1, "More than 1 language is not supported."

        super().__init__(
            columns=columns,
            vocab=vocab,
            tokenizer=tokenizer,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            use_eos_token_for_bos=use_eos_token_for_bos,
            wrap_special_tokens=wrap_special_tokens,
            max_seq_len=max_seq_len,
        )

        self.num_special_tokens = (1 if add_bos_token else 0) + (
            1 if add_eos_token else 0
        )
        self.language_columns = language_columns
        self.lang2id = lang2id
        self.use_language_embeddings = use_language_embeddings
        self.has_language_in_data = has_language_in_data

    @property
    def column_schema(self):
        schema = super().column_schema
        # input data always has a language for each instance
        if self.has_language_in_data:
            schema += [(column, str) for column in self.language_columns]
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
            return self.lang2id.get("en", 0)

    def _lookup_tokens(self, text: str, seq_len: int) -> List[str]:
        """
        XLM wraps every sentence with the EOS token both at the beginning and
        at the end. So the call to lookup_tokens can just use the flags as is.
        """
        return lookup_tokens(
            text,
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            bos_token=self.vocab.bos_token if self.add_bos_token else None,
            eos_token=self.vocab.eos_token if self.add_eos_token else None,
            use_eos_token_for_bos=self.use_eos_token_for_bos,
            max_seq_len=seq_len,
        )

    def wrap(self, numberized_sentences: List[List[str]]):
        raise NotImplementedError("XLM does not support special token wrapping.")

    def numberize(self, row: Dict) -> Tuple[Any, ...]:
        """
        Extract text and language information from the current row of data being
        processed. Process the text by tokenizing and vocabulary lookup. Convert
        language to corresponding list of ids.
        """
        sentences = []
        language_columns = self.language_columns
        columns = self.columns

        # sequence_length is adjusted based on the number of text fields and needs
        # to account for the special tokens which we will be wrapping
        for column in columns:
            sentences.extend(
                [
                    self._lookup_tokens(
                        text=row[column],
                        seq_len=(
                            self.max_seq_len // len(columns) - self.num_special_tokens
                        ),
                    )[0]
                ]
            )
        if self.wrap_special_tokens:
            sentences = self.wrap(sentences)
        tokens = list(itertools.chain(*sentences))
        seq_lens = [len(sentence) for sentence in sentences]
        lang_ids = [self.get_lang_id(row, language_columns[0])]
        if len(lang_ids) == 1:
            lang_ids = lang_ids * len(self.columns)

        # expand the language ids to each token
        lang_ids = ([lang_id] * seq_len for lang_id, seq_len in zip(lang_ids, seq_lens))
        segment_labels = list(itertools.chain(*lang_ids))
        seq_len = len(tokens)
        positions = [index for index in range(seq_len)]
        # return tokens, seq_len, segment_labels, positions
        return tokens, segment_labels, seq_len, positions

    def sort_key(self, row):
        return row[1]
