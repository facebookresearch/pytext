#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Set

import numpy as np
from pytext.common.constants import SpecialTokens, Token
from pytext.config.component import Component, ComponentType
from pytext.config.pytext_config import ConfigBase
from pytext.data.data_structures.annotation import Annotation, Intent, Root, Slot
from pytext.data.utils import VocabBuilder, Vocabulary


class MaskedVocabBuilder(VocabBuilder):
    def __init__(self, delimiter=" "):
        super().__init__(delimiter)
        self.use_mask = True


SPECIAL_TOKENS: Dict[str, Token] = {
    str(SpecialTokens.MASK): SpecialTokens.MASK,
    str(SpecialTokens.BOS): SpecialTokens.BOS,
    str(SpecialTokens.EOS): SpecialTokens.EOS,
}


class MaskingFunction(Component):
    class Config(ConfigBase):
        pass

    __EXPANSIBLE__ = True
    __COMPONENT_TYPE__ = ComponentType.MASKING_FUNCTION

    @classmethod
    def from_config(cls, config, use_bos, use_eos):
        return cls(use_bos, use_eos)

    def __init__(self, use_bos, use_eos):
        self.use_bos = use_bos
        self.use_eos = use_eos

    def should_mask(self, *args, **kwargs) -> bool:
        return True

    def gen_masked_source_target(self, tokens, *args, **kwargs):
        raise NotImplementedError()

    def _prepare_dec_target(
        self, dec_source: List[int], clean_input_tokens: List[int], vocab: Vocabulary
    ) -> List[int]:
        dec_target = [
            vocab.get_pad_index()
            if dec_source_token != vocab.get_mask_index()
            else dec_real_target_token
            for (dec_source_token, dec_real_target_token) in zip(
                dec_source, clean_input_tokens
            )
        ]

        return dec_target


class TreeMask(MaskingFunction):
    class Config(ConfigBase):
        accept_flat_intents_slots: bool = True
        factor: int = 2

    @classmethod
    def from_config(cls, config, use_bos, use_eos):
        return cls(config.accept_flat_intents_slots, config.factor, use_bos, use_eos)

    def __init__(self, accept_flat_intents_slots, factor, use_bos, use_eos):
        super().__init__(use_bos, use_eos)
        self.accept_flat_intents_slots = accept_flat_intents_slots
        self.factor = factor

    def clean_eos_bos(self, tokens):
        start_index, end_index = 0, len(tokens)
        if self.use_bos:
            start_index = 1
        if self.use_eos:
            end_index = -1
        return tokens[start_index:end_index]

    def gen_masked_tree(self, node, mask_token, depth=1):
        if self.should_mask(depth):
            actual_str_len = len(node.flat_str().strip().split(" "))
            return " ".join([mask_token for idx in range(actual_str_len)])
        else:
            return_str = " "
            if (
                isinstance(node, Intent)
                or isinstance(node, Slot)
                or isinstance(node, Root)
            ):
                return_str += "["
                return_str += node.label
                return_str += " "
                for child in node.children:
                    return_str += self.gen_masked_tree(child, mask_token, depth + 1)
                    return_str += " "
                return_str += "]"
            else:
                return_str += node.label
                return_str += " "
            return return_str.strip()

    def should_mask(self, depth=1):
        return np.random.random() < 1.0 / (self.factor ** depth)

    def gen_masked_source_target(self, tokens: List[int], vocab: Vocabulary):
        cleaned_tokens = self.clean_eos_bos(tokens)
        original_target_string = " ".join(
            [vocab[idx] for idx in cleaned_tokens]
        ).upper()
        try:
            annotation = Annotation(
                original_target_string,
                accept_flat_intents_slots=self.accept_flat_intents_slots,
            )
        except Exception as e:
            # This should never happen other than when testing
            print(e, original_target_string)
            dec_source = [vocab.idx[vocab.mask_token] for _ in range(len(tokens))]
            dec_target = [vocab.idx[vocab.pad_token] for _ in range(len(tokens))]
            return dec_source, dec_target
        assert len(annotation.root.children) == 1
        mask_tree_str = self.gen_masked_tree(
            annotation.root.children[0], vocab.mask_token
        )

        # We are calling the .split() instead of the tokenize() of tensorizer
        # because the input str contains special MASK token __MASK__
        # It we call tokenize() on this input_str, it may lower __MASK__ or split
        # in unexpected ways causing issues.
        # Hence temporary workaround is that we call split(" ") and lower all tokens
        # other than MASK tokens

        # handle special tokens in vocab
        mask_tree_str: List[str] = list(
            map(
                lambda token: SPECIAL_TOKENS.get(token, token.lower()),
                mask_tree_str.split(" "),
            )
        )

        dec_source = [vocab.idx.get(t) for t in mask_tree_str]

        dec_target = self._prepare_dec_target(dec_source, cleaned_tokens, vocab)

        if self.use_bos:
            if self.should_mask():
                dec_source.insert(0, vocab.get_mask_index())
                dec_target.insert(0, vocab.get_bos_index())
            else:
                dec_source.insert(0, vocab.get_bos_index())
                dec_target.insert(0, vocab.get_pad_index())

        if self.use_eos:
            if self.should_mask():
                dec_source.append(vocab.get_mask_index())
                dec_target.append(vocab.get_eos_index())
            else:
                dec_source.append(vocab.get_eos_index())
                dec_target.append(vocab.get_pad_index())
        return dec_source, dec_target


class MaskEverything(MaskingFunction):
    def gen_masked_tree(self, node, mask_token, depth=1):
        actual_str_len = len(node.flat_str().strip().split(" "))
        return " ".join([mask_token for idx in range(actual_str_len)])

    def gen_masked_source_target(self, tokens, vocab: Vocabulary):
        dec_source: List[int] = [vocab.get_mask_index() for idx in tokens]
        dec_target = self._prepare_dec_target(dec_source, tokens, vocab)
        return dec_source, dec_target


class RandomizedMaskingFunction(MaskingFunction):
    class Config(MaskingFunction.Config):
        seed: Optional[int] = None
        minimum_masks: int = 1

    @classmethod
    def from_config(cls, config: Config, use_bos: bool, use_eos: bool):
        return cls(config.seed, config.minimum_masks, use_bos, use_eos)

    def __init__(
        self, seed: Optional[int], minimum_masks: int, use_bos: bool, use_eos: bool
    ):
        super().__init__(use_bos, use_eos)
        self.random = np.random.RandomState(seed)
        self.minimum_masks = minimum_masks

    def gen_masked_source_target(self, tokens: List[int], vocab: Vocabulary):
        num_masks = self.random.randint(self.minimum_masks, len(tokens))

        ind: Set[int] = set(
            self.random.choice(len(tokens), size=num_masks, replace=False)
        )

        dec_source: List[int] = [
            vocab.get_mask_index() if idx in ind else token
            for idx, token in enumerate(tokens)
        ]

        dec_target = self._prepare_dec_target(dec_source, tokens, vocab)

        return dec_source, dec_target


class NoOpMaskingFunction(MaskingFunction):
    class Config(MaskingFunction.Config):
        seed: Optional[int] = None
        minimum_masks: int = 1

    @classmethod
    def from_config(cls, config: Config, use_bos: bool, use_eos: bool):
        return cls(config.seed, config.minimum_masks, use_bos, use_eos)

    def __init__(
        self, seed: Optional[int], minimum_masks: int, use_bos: bool, use_eos: bool
    ):
        super().__init__(use_bos, use_eos)
        self.random = np.random.RandomState(seed)
        self.minimum_masks = minimum_masks

    def gen_masked_source_target(self, tokens: List[int], vocab: Vocabulary):
        dec_target = self._prepare_dec_target(tokens, tokens, vocab)

        return tokens, dec_target
