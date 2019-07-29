#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
from typing import Any, Dict, List, Optional, Tuple

import torch
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from pytext.config.component import ComponentType, create_component
from pytext.data.bert_tensorizer import BERTTensorizer
from pytext.data.tokenizers import Tokenizer
from pytext.data.utils import BOS, EOS, MASK, PAD, UNK, Vocabulary, pad_and_tensorize
from pytext.data.xlm_dictionary import Dictionary as XLMDictionary


DEFAULT_LANG2ID_DICT = {
    "ar": 0,
    "bg": 1,
    "de": 2,
    "el": 3,
    "en": 4,
    "es": 5,
    "fr": 6,
    "hi": 7,
    "ru": 8,
    "sw": 9,
    "th": 10,
    "tr": 11,
    "ur": 12,
    "vi": 13,
    "zh": 14,
}


class XLMTensorizer(BERTTensorizer):
    """
    Tensorizer for Cross-lingual LM tasks. Works for single sentence as well
    as sentence pair.
    """

    class Config(BERTTensorizer.Config):
        vocab_file: str = "/mnt/vol/nlp_technologies/xlm/vocab_xnli_15"
        tokenizer: Tokenizer.Config = Tokenizer.Config()
        is_fairseq: bool = False
        pretraining: bool = False
        max_seq_len: Optional[int] = 256
        max_vocab: int = 95000
        min_count: int = 0
        language_columns: List[str] = ["language"]
        lang2id: Dict[str, int] = DEFAULT_LANG2ID_DICT
        reset_positions: bool = False
        has_language_in_data: bool = False
        use_language_embeddings: bool = True

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        return cls(
            columns=config.columns,
            tokenizer=tokenizer,
            vocab_file=config.vocab_file,
            is_fairseq=config.is_fairseq,
            pretraining=config.pretraining,
            max_seq_len=config.max_seq_len,
            max_vocab=config.max_vocab,
            min_count=config.min_count,
            language_columns=config.language_columns,
            lang2id=config.lang2id,
            reset_positions=config.reset_positions,
            has_language_in_data=config.has_language_in_data,
            use_language_embeddings=config.use_language_embeddings,
        )

    def __init__(
        self,
        columns=Config.columns,
        tokenizer=None,
        vocab_file=Config.vocab_file,
        is_fairseq=Config.is_fairseq,
        pretraining=Config.pretraining,
        max_seq_len=Config.max_seq_len,
        max_vocab=Config.max_vocab,
        min_count=Config.min_count,
        language_columns=Config.language_columns,
        lang2id=Config.lang2id,
        reset_positions=Config.reset_positions,
        has_language_in_data=Config.has_language_in_data,
        use_language_embeddings=Config.use_language_embeddings,
    ) -> None:
        assert (
            len(columns) <= 2 and len(language_columns) <= 2
        ), "Number of text fields and language columns cannot be greater than 2."

        assert (
            len(language_columns) == len(columns) or len(language_columns) == 1
        ), "language columns must be same length as columns, or of length one"

        # Used to distinguish the model pre-trained in PyText and the OSS FAIR model
        self.is_fairseq = is_fairseq

        # controls the settings we need explictly for pretraining
        self.pretraining = pretraining

        # language identifiers for extracting the language from a row of data
        # during numberize
        self.language_columns = language_columns

        # language-to-id mapping used to obtain language embeddings
        self.lang2id = lang2id

        vocab = self._build_vocab(vocab_file, max_vocab, min_count)
        self.special_token = vocab.idx[EOS]

        # Controls whether we reset the position or not in case we have
        # multiplt text fields
        self.reset_positions = reset_positions

        # controls whether we train with language embeddings or not
        self.use_language_embeddings = use_language_embeddings

        super().__init__(
            columns=columns,
            tokenizer=tokenizer,
            add_bos_token=False,
            add_eos_token=False,
            use_eos_token_for_bos=True,
            vocab=vocab,
            max_seq_len=max_seq_len,
        )
        # if the dataset has a language column then adjust the schema appropriately
        self.has_language_in_data = has_language_in_data

    @property
    def column_schema(self):
        schema = super().column_schema
        if self.has_language_in_data:
            schema += [(column, str) for column in self.language_columns]
        return schema

    def _lookup_tokens(self, text: str, seq_len: int) -> List[int]:
        tokenized_text = [t.value for t in self.tokenizer.tokenize(text)]
        truncated_text = tokenized_text[:seq_len]
        tokens = self.vocab.lookup_all(truncated_text)
        return tokens

    def _numberize_and_wrap(self, text: str, seq_len: int) -> List[List[int]]:
        sentence = (
            [self.special_token]
            + self._lookup_tokens(text, seq_len)
            + [self.special_token]
        )
        return [sentence]

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

    def numberize(self, row: Dict) -> Tuple[Any, ...]:
        """
        Extract text and language information from the current row of data being
        processed. Process the text by tokenizing and vocabulary lookup. Convert
        language to corresponding list of ids. This function can handle both
        monolingual and parallel data as well as single sentence and sentence
        pairs.
        """
        sentences = []
        language_columns = self.language_columns
        columns = self.columns

        # When we have sentence pairs, we need to truncate each text field to
        # max_seq_len // 2. However, the one case where this is not true
        # is the case when we're pre-training MLM + TLM and the given row
        # corresponds to monolingual data. Here the target text
        # (text field related to columns[1]) is None and we don't need to adjust
        # sequence length. Instead, we update columns and language_columns for
        # this row. The following if statement handels this case.
        if self.pretraining and len(columns) == 2:
            if row[columns[1]] is None:
                columns = columns[:1]
                language_columns = ["language"]

        # sequence_length is adjusted based on the number of text fields and needs
        # to account for the special tokens which we will be wrapping
        for column in columns:
            sentences.extend(
                self._numberize_and_wrap(
                    text=row[column], seq_len=(self.max_seq_len // len(columns) - 2)
                )
            )

        tokens = list(itertools.chain(*sentences))
        seq_lens = [len(sentence) for sentence in sentences]

        # create the language tensor. if only one language column is specified,
        # use it for all texts
        lang_ids = [self.get_lang_id(row, column) for column in language_columns]
        if len(lang_ids) == 1:
            lang_ids = lang_ids * len(self.columns)

        # expand the language ids to each token
        lang_ids = ([lang_id] * seq_len for lang_id, seq_len in zip(lang_ids, seq_lens))
        lang_ids = list(itertools.chain(*lang_ids))

        length = len(tokens)

        if not self.reset_positions:
            positions = [index for index in range(length)]
        else:
            positions = (range(seq_len) for seq_len in seq_lens)
            positions = list(itertools.chain(*positions))
        return tokens, length, lang_ids, positions

    def sort_key(self, row):
        return row[1]

    def tensorize(self, batch) -> Tuple[torch.Tensor, ...]:
        tokens, seq_lens, lang_ids, positions = zip(*batch)
        padded_tokens = pad_and_tensorize(tokens, self.vocab.get_pad_index())
        padded_lang_ids = pad_and_tensorize(lang_ids)
        if self.is_fairseq:
            positions = pad_and_tensorize(positions)
        else:
            positions = None
        pad_mask = (padded_tokens != self.vocab.get_pad_index()).long()
        return (
            padded_tokens,
            pad_mask,
            pad_and_tensorize(seq_lens),
            padded_lang_ids,
            positions,
        )

    def _read_vocab(
        self, vocab_file: str, max_vocab: int, min_count: int
    ) -> Tuple[List, List, Dict]:
        dictionary = XLMDictionary.read_vocab(vocab_file)
        if max_vocab >= 1:
            dictionary.max_vocab(max_vocab)
        if min_count >= 0:
            dictionary.min_count(min_count)
        vocab_list = [dictionary.id2word[w] for w in sorted(dictionary.id2word)]
        counts = [dictionary.counts[w] for w in vocab_list]
        replacements = {"<unk>": UNK, "<pad>": PAD, "<s>": BOS, "</s>": EOS}
        return vocab_list, counts, replacements

    def _read_fairseq_vocab(
        self, vocab_file: str, max_vocab: int = -1, min_count: int = -1
    ) -> Tuple[List, List, Dict]:
        dictionary = MaskedLMDictionary.load(vocab_file)
        dictionary.finalize(threshold=min_count, nwords=max_vocab, padding_factor=1)
        vocab_list = dictionary.symbols
        counts = dictionary.count
        replacements = {"<pad>": PAD, "</s>": EOS, "<unk>": UNK, "<mask>": MASK}
        return vocab_list, counts, replacements

    def _build_vocab(
        self, vocab_file: str, max_vocab: int, min_count: int
    ) -> Vocabulary:
        """
        Build Vocab for XLM by calling the vocab reader associated with the model
        source.
        """
        if self.is_fairseq:
            vocab_list, counts, replacements = self._read_fairseq_vocab(
                vocab_file, max_vocab, min_count
            )
        else:
            vocab_list, counts, replacements = self._read_vocab(
                vocab_file, max_vocab, min_count
            )
        return Vocabulary(vocab_list, counts, replacements=replacements)
