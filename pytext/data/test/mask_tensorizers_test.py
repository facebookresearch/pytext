#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import types
import unittest
from typing import List

from pytext.data.masked_tensorizer import MaskedTokenTensorizer
from pytext.data.masked_util import MaskEverything, RandomizedMaskingFunction, TreeMask
from pytext.data.sources.data_source import SafeFileWrapper
from pytext.data.sources.tsv import TSVDataSource
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class MaskTensorizersTest(unittest.TestCase):
    def setUp(self):
        self.data = TSVDataSource(
            SafeFileWrapper(tests_module.test_file("compositional_seq2seq_unit.tsv")),
            test_file=None,
            eval_file=None,
            field_names=["text", "seqlogical"],
            schema={"text": str, "seqlogical": str},
        )
        self.masked_tensorizer = MaskedTokenTensorizer.from_config(
            MaskedTokenTensorizer.Config(
                column="seqlogical", masking_function=TreeMask.Config()
            )
        )
        self._initialize_tensorizer(self.masked_tensorizer)

    def _initialize_tensorizer(self, tensorizer, data=None):
        if data is None:
            data = self.data
        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in data.train:
            init.send(row)
        init.close()

    def test_basic_tree_masking(self):

        rows = [
            {
                "text": "delays in tempe",
                "seqlogical": "[in:get_info_traffic delays in [sl:location tempe ] ]",
            },
            {
                "text": "find me the quickest route home",
                "seqlogical": "[in:get_directions find me the quickest route [sl:destination [in:get_location_home home ] ] ]",
            },
        ]

        vocab = self.masked_tensorizer.vocab
        masked_results = self.masked_tensorizer.tensorize(
            [self.masked_tensorizer.numberize(row) for row in rows]
        )

        all_tokens, _, _, all_masked_source, all_masked_target = masked_results
        for tokens, masked_source, masked_target in zip(
            all_tokens, all_masked_source, all_masked_target
        ):
            assert len(masked_source) == len(masked_target)
            assert len(tokens) == len(masked_target)
            for i in range(len(masked_source)):
                # For masked tokens, dec_target is real target tokens
                if masked_source[i] == vocab.get_mask_index():
                    assert masked_target[i] == tokens[i], (
                        str(masked_target[i]) + " != " + str(tokens[i])
                    )
                # For unmasked, target is pad token
                elif masked_source[i] != vocab.get_mask_index():
                    assert masked_target[i] == vocab.get_pad_index()

    def test_mask_at_depth_k(self):
        rows = [
            {
                "text": "find me the quickest route home",
                "seqlogical": "[in:get_directions find me the quickest route [sl:destination [in:get_location_home home ] ] ]",
            }
        ]

        vocab = self.masked_tensorizer.vocab

        def should_mask(self, depth=1):
            if depth == 3:
                return True
            else:
                return False

        self.masked_tensorizer.mask.should_mask = types.MethodType(
            should_mask, self.masked_tensorizer.mask
        )
        masked_results = self.masked_tensorizer.tensorize(
            [self.masked_tensorizer.numberize(row) for row in rows]
        )

        all_tokens, _, _, all_masked_source, _all_masked_target = masked_results
        _, masked_source = (all_tokens[0], all_masked_source[0])
        masked_source_tokens: List[str] = [vocab[tok] for tok in masked_source]
        self.assertEqual(
            masked_source_tokens,
            [
                "[in:get_directions",
                "find",
                "me",
                "the",
                "quickest",
                "route",
                "[sl:destination",
                vocab[vocab.get_mask_index()],
                vocab[vocab.get_mask_index()],
                vocab[vocab.get_mask_index()],
                "]",
                "]",
            ],
        )

    def test_tree_mask_with_bos_eos(self):
        rows = [
            {
                "text": "find me the quickest route home",
                "seqlogical": "[in:get_directions find me the quickest route [sl:destination [in:get_location_home home ] ] ]",
            }
        ]

        masked_tensorizer = MaskedTokenTensorizer.from_config(
            MaskedTokenTensorizer.Config(
                column="seqlogical",
                masking_function=TreeMask.Config(),
                add_bos_token=True,
                add_eos_token=True,
            )
        )

        self._initialize_tensorizer(masked_tensorizer)

        vocab = masked_tensorizer.vocab

        def should_mask(self, depth=1):
            if depth == 3:
                return True
            else:
                return False

        masked_tensorizer.mask.should_mask = types.MethodType(
            should_mask, masked_tensorizer.mask
        )
        masked_results = masked_tensorizer.tensorize(
            [masked_tensorizer.numberize(row) for row in rows]
        )

        all_tokens, _, _, all_masked_source, _all_masked_target = masked_results
        _, masked_source = (all_tokens[0], all_masked_source[0])
        masked_source_tokens: List[str] = [vocab[tok] for tok in masked_source]
        self.assertEqual(
            masked_source_tokens,
            [
                vocab.bos_token,
                "[in:get_directions",
                "find",
                "me",
                "the",
                "quickest",
                "route",
                "[sl:destination",
                vocab.mask_token,
                vocab.mask_token,
                vocab.mask_token,
                "]",
                "]",
                vocab.eos_token,
            ],
        )

    def test_mask_all(self):
        rows = [
            {
                "text": "find me the quickest route home",
                "seqlogical": "[in:get_directions find me the quickest route [sl:destination [in:get_location_home home ] ] ]",
            }
        ]

        masked_tensorizer = MaskedTokenTensorizer.from_config(
            MaskedTokenTensorizer.Config(
                column="seqlogical", masking_function=MaskEverything.Config()
            )
        )

        self._initialize_tensorizer(masked_tensorizer)

        vocab = masked_tensorizer.vocab
        masked_results = masked_tensorizer.tensorize(
            [masked_tensorizer.numberize(row) for row in rows]
        )

        all_tokens, _, _, all_masked_source, _all_masked_target = masked_results
        _, masked_source = (all_tokens[0], all_masked_source[0])
        masked_tokens: List[str] = [vocab[tok] for tok in masked_source]
        self.assertEqual(
            masked_tokens, [vocab[vocab.get_mask_index()]] * len(masked_tokens)
        )

    def test_mask_random(self):
        rows = [
            {
                "text": "find me the quickest route home",
                "seqlogical": "[in:get_directions find me the quickest route [sl:destination [in:get_location_home home ] ] ]",
            }
        ]

        masked_tensorizer = MaskedTokenTensorizer.from_config(
            MaskedTokenTensorizer.Config(
                column="seqlogical",
                masking_function=RandomizedMaskingFunction.Config(seed=2),
            )
        )

        self._initialize_tensorizer(masked_tensorizer)

        vocab = masked_tensorizer.vocab
        masked_results = masked_tensorizer.tensorize(
            [masked_tensorizer.numberize(row) for row in rows]
        )

        all_tokens, _, _, all_masked_source, _all_masked_target = masked_results
        _, masked_source = (all_tokens[0], all_masked_source[0])
        masked_tokens: List[str] = [vocab[tok] for tok in masked_source]
        target: List[str] = [
            vocab[vocab.get_mask_index()],
            vocab[vocab.get_mask_index()],
            "me",
            vocab[vocab.get_mask_index()],
            vocab[vocab.get_mask_index()],
            vocab[vocab.get_mask_index()],
            "[sl:destination",
            vocab[vocab.get_mask_index()],
            "home",
            vocab[vocab.get_mask_index()],
            vocab[vocab.get_mask_index()],
            vocab[vocab.get_mask_index()],
        ]
        self.assertEqual(masked_tokens, target)

    def test_mask_no_op(self):
        rows = [
            {
                "text": "find me the quickest route home",
                "seqlogical": "[in:get_directions find me the quickest route [sl:destination [in:get_location_home home ] ] ]",
            }
        ]

        masked_tensorizer = MaskedTokenTensorizer.from_config(
            MaskedTokenTensorizer.Config(
                column="seqlogical",
                masking_function=RandomizedMaskingFunction.Config(seed=2),
            )
        )

        self._initialize_tensorizer(masked_tensorizer)

        vocab = masked_tensorizer.vocab
        masked_results = masked_tensorizer.tensorize(
            [masked_tensorizer.numberize(row) for row in rows]
        )

        all_tokens, _, _, all_masked_source, _all_masked_target = masked_results
        _, masked_source = (all_tokens[0], all_masked_source[0])
        masked_tokens: List[str] = [vocab[tok] for tok in masked_source]
        target: List[str] = [
            vocab[vocab.get_mask_index()],
            vocab[vocab.get_mask_index()],
            "me",
            vocab[vocab.get_mask_index()],
            vocab[vocab.get_mask_index()],
            vocab[vocab.get_mask_index()],
            "[sl:destination",
            vocab[vocab.get_mask_index()],
            "home",
            vocab[vocab.get_mask_index()],
            vocab[vocab.get_mask_index()],
            vocab[vocab.get_mask_index()],
        ]
        self.assertEqual(masked_tokens, target)
