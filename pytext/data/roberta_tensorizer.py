#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
from typing import Any, Dict, List, Tuple

import torch
from pytext import resources
from pytext.common.constants import SpecialTokens
from pytext.config.component import ComponentType, create_component
from pytext.data.bert_tensorizer import (
    BERTTensorizerBase,
    BERTTensorizerBaseScriptImpl,
    build_fairseq_vocab,
)
from pytext.data.tokenizers import GPT2BPETokenizer, Tokenizer
from pytext.data.utils import VocabBuilder, pad_and_tensorize
from pytext.torchscript.tensorizer import ScriptRoBERTaTensorizer
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils.file_io import PathManager


RoBERTaTensorizerScriptImpl = BERTTensorizerBaseScriptImpl


class RoBERTaTensorizer(BERTTensorizerBase):

    __TENSORIZER_SCRIPT_IMPL__ = RoBERTaTensorizerScriptImpl

    class Config(BERTTensorizerBase.Config):
        # any unittest should be overriding this with a small local file
        vocab_file: str = resources.roberta.GPT2_BPE_DICT
        tokenizer: Tokenizer.Config = GPT2BPETokenizer.Config()
        max_seq_len: int = 256

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        base_tokenizer = None
        if config.base_tokenizer:
            base_tokenizer = create_component(
                ComponentType.TOKENIZER, config.base_tokenizer
            )

        # map to the real vocab_file
        config.vocab_file = (
            resources.roberta.RESOURCE_MAP[config.vocab_file]
            if config.vocab_file in resources.roberta.RESOURCE_MAP
            else config.vocab_file
        )
        with PathManager.open(config.vocab_file) as f:
            vocab = build_fairseq_vocab(
                vocab_file=f,
                special_token_replacements={
                    "<pad>": SpecialTokens.PAD,
                    "<s>": SpecialTokens.BOS,
                    "</s>": SpecialTokens.EOS,
                    "<unk>": SpecialTokens.UNK,
                    "<mask>": SpecialTokens.MASK,
                },
            )
        return cls(
            columns=config.columns,
            vocab=vocab,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            base_tokenizer=base_tokenizer,
        )


class RoBERTaTokenLevelTensorizer(RoBERTaTensorizer):
    """
    Tensorizer for token level classification tasks such as NER, POS etc
    using RoBERTa. Here each token has an associated label and the tensorizer
    should output a label tensor as well. The input for this tensorizer comes
    from the CoNLLUNERDataSource data source.
    """

    class Config(RoBERTaTensorizer.Config):
        # label identifiers for extracting the label from a row of data
        # during numberize and for modifying the schema
        labels_columns: List[str] = ["label"]
        # label space for the task
        labels: List[str] = []

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        with PathManager.open(config.vocab_file) as f:
            vocab = build_fairseq_vocab(
                vocab_file=f,
                special_token_replacements={
                    "<pad>": SpecialTokens.PAD,
                    "<s>": SpecialTokens.BOS,
                    "</s>": SpecialTokens.EOS,
                    "<unk>": SpecialTokens.UNK,
                    "<mask>": SpecialTokens.MASK,
                },
            )
        return cls(
            columns=config.columns,
            vocab=vocab,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            labels_columns=config.labels_columns,
            labels=config.labels,
        )

    def __init__(
        self,
        columns,
        tokenizer=None,
        vocab=None,
        max_seq_len=256,
        labels_columns=Config.labels_columns,
        labels=Config.labels,
    ) -> None:
        assert len(columns) == 1, "Only support single sentences for token level tasks."
        assert len(labels) >= 1, "Label set should not be empty."
        super().__init__(
            columns=columns, tokenizer=tokenizer, max_seq_len=max_seq_len, vocab=vocab
        )
        self.labels = labels
        self.labels_columns = labels_columns
        self.labels_vocab = self._build_label_vocab(labels)
        self.labels_pad_idx = self.labels_vocab.get_pad_index()

    def _build_label_vocab(self, labels: List[str]):
        labels_vocab_builder = VocabBuilder()
        labels_vocab_builder.use_pad = True
        labels_vocab_builder.use_unk = False

        labels_vocab_builder.add_all(labels)
        return labels_vocab_builder.make_vocab()

    @property
    def column_schema(self):
        schema = super().column_schema
        schema += [(self.labels_columns[0], str)]
        return schema

    def numberize(self, row: Dict) -> Tuple[Any, ...]:
        """
        Numberize both the tokens and labels. Since we break up tokens, the label
        for anything other than the first sub-word is assigned the padding idx.
        """
        # The relevant data source outputs a list of strings for both
        # the tokens and labels
        raw_tokens = row[self.columns[0]]
        raw_labels = row[self.labels_columns[0]]

        numberized_tokens = []
        numberized_labels = []

        for token, label in zip(raw_tokens, raw_labels):
            sub_tokens = self.tokenizer.tokenize(token)

            # convert token level labels to sub-word level labels
            for i in range(len(sub_tokens)):
                sub_token_id = self.vocab.lookup_all(sub_tokens[i].value)
                label_id = (
                    self.labels_vocab.lookup_all(label)
                    if i == 0
                    else self.labels_pad_idx
                )

                numberized_tokens.append(sub_token_id)
                numberized_labels.append(label_id)

        # Ensure the numberized labels and tokens are wrapped correctly
        numberized_sentences = [
            [self.vocab.get_bos_index()]
            + numberized_tokens[: self.max_seq_len - 2]
            + [self.vocab.get_eos_index()]
        ]
        numberized_labels = (
            [self.labels_pad_idx]
            + numberized_labels[: self.max_seq_len - 2]
            + [self.labels_pad_idx]
        )

        seq_lens = (len(sentence) for sentence in numberized_sentences)
        segment_labels = ([i] * seq_len for i, seq_len in enumerate(seq_lens))
        tokens = list(itertools.chain(*numberized_sentences))
        segment_labels = list(itertools.chain(*segment_labels))
        seq_len = len(tokens)
        positions = [index for index in range(seq_len)]
        return tokens, segment_labels, seq_len, positions, numberized_labels

    def tensorize(self, batch) -> Tuple[torch.Tensor, ...]:
        tokens, segment_labels, seq_lens, positions, labels = zip(*batch)
        tokens = pad_and_tensorize(tokens, self.vocab.get_pad_index())
        pad_mask = (tokens != self.vocab.get_pad_index()).long()
        segment_labels = pad_and_tensorize(segment_labels, self.vocab.get_pad_index())
        positions = pad_and_tensorize(positions)
        padded_labels = pad_and_tensorize(labels, self.labels_pad_idx)
        return tokens, pad_mask, segment_labels, positions, padded_labels

    def torchscriptify(self):
        return ScriptRoBERTaTensorizer(
            tokenizer=self.tokenizer.torchscriptify(),
            vocab=ScriptVocabulary(
                list(self.vocab),
                pad_idx=self.vocab.get_pad_index(),
                bos_idx=self.vocab.get_bos_index(),
                eos_idx=self.vocab.get_eos_index(),
            ),
            max_seq_len=self.max_seq_len,
        )
