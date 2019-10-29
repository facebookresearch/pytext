#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

from fairseq.data.dictionary import Dictionary
from pytext.config.component import ComponentType, create_component
from pytext.data.bert_tensorizer import BERTTensorizer, build_fairseq_vocab
from pytext.data.tensorizers import Tensorizer, lookup_tokens
from pytext.data.tokenizers import Gpt2Tokenizer, Tokenizer
from pytext.data.utils import BOS, EOS, PAD, UNK, Vocabulary
from pytext.torchscript.tensorizer import ScriptRoBERTaTensorizer
from pytext.utils.torch import Vocabulary as ScriptVocabulary


def build_roberta_vocab(
    pad_token: str, bos_token: str, eos_token: str, unk_token: str, vocab_file: str
) -> Vocabulary:
    special_token_replacements = {
        pad_token: PAD,
        bos_token: BOS,
        eos_token: EOS,
        unk_token: UNK,
    }
    vocab = build_fairseq_vocab(
        dictionary_class=Dictionary,
        vocab_file=vocab_file,
        special_token_replacements=special_token_replacements,
    )
    return vocab


class RoBERTaTensorizer(BERTTensorizer):
    class Config(Tensorizer.Config):
        columns: List[str] = ["text"]
        vocab_file: str = (
            "manifold://pytext_training/tree/static/vocabs/bpe/gpt2/dict.txt"
        )
        tokenizer: Tokenizer.Config = Gpt2Tokenizer.Config()
        bos_token: str = "<s>"
        eos_token: str = "</s>"
        pad_token: str = "<pad>"
        unk_token: str = "<unk>"
        max_seq_len: int = 256

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        vocab = build_roberta_vocab(
            config.pad_token,
            config.bos_token,
            config.eos_token,
            config.unk_token,
            config.vocab_file,
        )
        return cls(
            columns=config.columns,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            vocab=vocab,
        )

    def __init__(self, columns, tokenizer=None, vocab=None, max_seq_len=256):
        super().__init__(
            columns=columns,
            tokenizer=tokenizer,
            add_bos_token=False,
            add_eos_token=True,
            max_seq_len=max_seq_len,
            vocab=vocab,
        )

    def _lookup_tokens(self, text):
        return lookup_tokens(
            text,
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            bos_token=self.vocab.bos_token,
            eos_token=self.vocab.eos_token,
            max_seq_len=self.max_seq_len,
        )

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
            add_bos_token=True,
            use_eos_token_for_bos=False,
        )
