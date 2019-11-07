#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

from pytext.config.component import ComponentType, create_component
from pytext.data.bert_tensorizer import BERTTensorizer, build_fairseq_vocab
from pytext.data.tensorizers import Tensorizer, lookup_tokens
from pytext.data.tokenizers import GPT2BPETokenizer, Tokenizer
from pytext.data.utils import BOS, EOS, PAD, UNK, Vocabulary
from pytext.torchscript.tensorizer import (
    ScriptRoBERTaTensorizer,
    ScriptRoBERTaTokenTensorizer,
)
from pytext.torchscript.vocab import ScriptVocabulary


class RoBERTaTensorizer(BERTTensorizer):
    class Config(Tensorizer.Config):
        columns: List[str] = ["text"]
        vocab_file: str = (
            "manifold://pytext_training/tree/static/vocabs/bpe/gpt2/dict.txt"
        )
        tokenizer: GPT2BPETokenizer.Config = GPT2BPETokenizer.Config()
        # Make special tokens configurable so we don't need a new
        # tensorizer if the model is trained with different special token
        bos_token: str = "<s>"
        eos_token: str = "</s>"
        pad_token: str = "<pad>"
        unk_token: str = "<unk>"
        max_seq_len: int = 256

    @classmethod
    def from_config(cls, config: Config, **kwargs):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        vocab = build_fairseq_vocab(
            vocab_file=config.vocab_file,
            special_token_replacements={
                config.pad_token: PAD,
                config.bos_token: BOS,
                config.eos_token: EOS,
                config.unk_token: UNK,
            },
        )
        return cls(
            columns=config.columns,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            vocab=vocab,
        )

    def __init__(
        self,
        columns: List[str],
        tokenizer: Tokenizer = None,
        vocab: Vocabulary = None,
        max_seq_len=256,
    ) -> None:
        super().__init__(
            columns=columns,
            tokenizer=tokenizer,
            add_bos_token=False,
            add_eos_token=True,
            max_seq_len=max_seq_len,
            vocab=vocab,
        )

    def _lookup_tokens(self, text: str):
        return lookup_tokens(
            text,
            tokenizer=self.tokenizer,
            vocab=self.vocab,
            bos_token=self.vocab.bos_token,
            eos_token=self.vocab.eos_token,
            max_seq_len=self.max_seq_len,
        )

    def torchscriptify(self):
        input_type = self.tokenizer.torchscriptify_input_type()
        if input_type.is_text():
            script_tensorizer_class = ScriptRoBERTaTensorizer
        elif input_type.is_token():
            script_tensorizer_class = ScriptRoBERTaTokenTensorizer
        else:
            raise RuntimeError(f"Unsupported {input_type}...")

        return script_tensorizer_class(
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
