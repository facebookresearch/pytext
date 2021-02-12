#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from pytext.config.component import ComponentType, create_component
from pytext.data.masked_util import (
    MaskingFunction,
    RandomizedMaskingFunction,
    MaskedVocabBuilder,
)
from pytext.data.tensorizers import TokenTensorizer
from pytext.data.utils import pad, pad_and_tensorize
from pytext.utils import cuda


class MaskedTokenTensorizer(TokenTensorizer):
    class Config(TokenTensorizer.Config):
        masking_function: MaskingFunction.Config = RandomizedMaskingFunction.Config()

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        mask = create_component(
            ComponentType.MASKING_FUNCTION,
            config.masking_function,
            config.add_bos_token,
            config.add_eos_token,
        )
        return cls(
            text_column=config.column,
            mask=mask,
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
        mask,
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
        super().__init__(
            text_column,
            tokenizer,
            add_bos_token,
            add_eos_token,
            use_eos_token_for_bos,
            max_seq_len,
            vocab_config,
            vocab,
            vocab_file_delimiter,
            is_input,
        )
        self.mask = mask
        self.vocab_builder = MaskedVocabBuilder()
        self.vocab_builder.use_bos = add_bos_token
        self.vocab_builder.use_eos = add_eos_token

    def mask_and_tensorize(self, batch):
        batch = list(batch)
        if not batch:
            return torch.Tensor()
        masked_sources = []
        masked_targets = []

        for tokens in batch:
            dec_source, dec_target = self.mask.gen_masked_source_target(
                tokens, vocab=self.vocab
            )

            masked_sources.append(dec_source)
            masked_targets.append(dec_target)

        return (
            cuda.tensor(
                pad(masked_sources, pad_token=self.vocab.get_pad_index()),
                dtype=torch.long,
            ),
            cuda.tensor(
                pad(masked_targets, pad_token=self.vocab.get_pad_index()),
                dtype=torch.long,
            ),
        )

    def tensorize(self, batch):
        tokens, seq_lens, token_ranges = zip(*batch)
        masked_source, masked_target = self.mask_and_tensorize(tokens)
        return (
            pad_and_tensorize(tokens, self.vocab.get_pad_index()),
            pad_and_tensorize(seq_lens),
            pad_and_tensorize(token_ranges),
            masked_source,
            masked_target,
        )
