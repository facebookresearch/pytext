#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from pytext.config.component import ComponentType, create_component
from pytext.data.bert_tensorizer import BERTTensorizerBase, build_fairseq_vocab
from pytext.data.tokenizers import GPT2BPETokenizer, Tokenizer
from pytext.data.utils import BOS, EOS, MASK, PAD, UNK
from pytext.torchscript.tensorizer import ScriptRoBERTaTensorizer
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils.file_io import PathManager


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
        base_tokenizer = None
        if config.base_tokenizer:
            base_tokenizer = create_component(
                ComponentType.TOKENIZER, config.base_tokenizer
            )
        with PathManager.open(config.vocab_file) as f:
            vocab = build_fairseq_vocab(
                vocab_file=f,
                special_token_replacements={
                    "<pad>": PAD,
                    "<s>": BOS,
                    "</s>": EOS,
                    "<unk>": UNK,
                    "<mask>": MASK,
                },
            )
        return cls(
            columns=config.columns,
            vocab=vocab,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
            base_tokenizer=base_tokenizer,
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
        )
