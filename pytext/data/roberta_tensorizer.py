#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

from pytext.config.component import ComponentType, create_component
from pytext.data.bert_tensorizer import BERTTensorizerBase, build_fairseq_vocab
from pytext.data.tokenizers import GPT2BPETokenizer, Tokenizer
from pytext.data.utils import BOS, EOS, PAD, UNK, Vocabulary
from pytext.torchscript.tensorizer import (
    ScriptRoBERTaTensorizer,
    ScriptRoBERTaTokenTensorizer,
)
from pytext.torchscript.vocab import ScriptVocabulary


class RoBERTaTensorizer(BERTTensorizerBase):
    class Config(BERTTensorizerBase.Config):
        vocab_file: str = (
            "manifold://pytext_training/tree/static/vocabs/bpe/gpt2/dict.txt"
        )
        tokenizer: Tokenizer.Config = GPT2BPETokenizer.Config()
        max_seq_len: int = 256

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        vocab = build_fairseq_vocab(
            vocab_file=config.vocab_file,
            special_token_replacements={
                "<pad>": PAD,
                "<s>": BOS,
                "</s>": EOS,
                "<unk>": UNK,
            },
        )
        return cls(
            columns=config.columns,
            vocab=vocab,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
        )

    def __init__(
        self,
        columns: List[str] = Config.columns,
        vocab: Vocabulary = None,
        tokenizer: Tokenizer = None,
        max_seq_len: int = Config.max_seq_len,
    ) -> None:
        super().__init__(
            columns=columns, vocab=vocab, tokenizer=tokenizer, max_seq_len=max_seq_len
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
