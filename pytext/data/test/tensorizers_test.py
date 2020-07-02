#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
from typing import List

import numpy as np
import pandas as pd
import torch
from pytext.common.constants import SpecialTokens
from pytext.data.bert_tensorizer import BERTTensorizer, BERTTensorizerScriptImpl
from pytext.data.roberta_tensorizer import (
    RoBERTaTensorizer,
    RoBERTaTensorizerScriptImpl,
)
from pytext.data.sources import SquadDataSource
from pytext.data.sources.data_source import Gazetteer, SafeFileWrapper, load_float_list
from pytext.data.sources.pandas import SessionPandasDataSource
from pytext.data.sources.tsv import SessionTSVDataSource, TSVDataSource
from pytext.data.squad_for_bert_tensorizer import (
    SquadForBERTTensorizer,
    SquadForRoBERTaTensorizer,
)
from pytext.data.squad_tensorizer import SquadTensorizer
from pytext.data.tensorizers import (
    AnnotationNumberizer,
    ByteTensorizer,
    ByteTokenTensorizer,
    FloatListSeqTensorizer,
    FloatListTensorizer,
    GazetteerTensorizer,
    LabelListTensorizer,
    LabelTensorizer,
    SeqTokenTensorizer,
    TokenTensorizer,
    VocabConfig,
    VocabFileConfig,
    initialize_tensorizers,
    lookup_tokens,
)
from pytext.data.token_tensorizer import ScriptBasedTokenTensorizer
from pytext.data.tokenizers import (
    DoNothingTokenizer,
    GPT2BPETokenizer,
    Tokenizer,
    WordPieceTokenizer,
)
from pytext.data.utils import Vocabulary
from pytext.utils import precision
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class LookupTokensTest(unittest.TestCase):
    def test_lookup_tokens(self):
        text = "let's tokenize this"
        tokenizer = Tokenizer()
        vocab = Vocabulary(text.split() + [SpecialTokens.BOS, SpecialTokens.EOS])
        tokens, start_idx, end_idx = lookup_tokens(
            text, tokenizer=tokenizer, vocab=vocab, bos_token=None, eos_token=None
        )
        self.assertEqual(tokens, [0, 1, 2])
        self.assertEqual(start_idx, (0, 6, 15))
        self.assertEqual(end_idx, (5, 14, 19))

        tokens, start_idx, end_idx = lookup_tokens(
            text,
            tokenizer=tokenizer,
            vocab=vocab,
            bos_token=SpecialTokens.BOS,
            eos_token=SpecialTokens.EOS,
        )
        self.assertEqual(tokens, [3, 0, 1, 2, 4])
        self.assertEqual(start_idx, (-1, 0, 6, 15, -1))
        self.assertEqual(end_idx, (-1, 5, 14, 19, -1))


class ListTensorizersTest(unittest.TestCase):
    def setUp(self):
        self.data = SessionTSVDataSource(
            SafeFileWrapper(tests_module.test_file("seq_tagging_example.tsv")),
            field_names=["session_id", "intent", "goal", "label"],
            schema={"intent": List[str], "goal": List[str], "label": List[str]},
        )

    def test_initialize_list_tensorizers(self):
        tensorizers = {
            "intent": LabelListTensorizer(
                label_column="intent", pad_in_vocab=True, allow_unknown=True
            ),
            "goal": LabelListTensorizer(label_column="goal"),
        }
        initialize_tensorizers(tensorizers, self.data.train)
        self.assertEqual(9, len(tensorizers["intent"].vocab))
        self.assertEqual(7, len(tensorizers["goal"].vocab))

    def test_create_label_list_tensors(self):
        tensorizers = {
            "intent": LabelListTensorizer(
                label_column="intent", pad_in_vocab=True, allow_unknown=True
            )
        }
        initialize_tensorizers(tensorizers, self.data.train)
        tensors = [tensorizers["intent"].numberize(row) for row in self.data.train]
        # test label idx
        self.assertEqual([2, 3], tensors[0][0])
        self.assertEqual([4, 5], tensors[1][0])
        self.assertEqual([6, 7, 8], tensors[2][0])
        # test seq lens
        self.assertEqual(2, tensors[0][1])
        self.assertEqual(2, tensors[1][1])
        self.assertEqual(3, tensors[2][1])
        self.assertEqual(3, len(tensors))
        tensors, lens = tensorizers["intent"].tensorize(tensors)
        np.testing.assert_array_almost_equal(
            np.array([[2, 3, 1], [4, 5, 1], [6, 7, 8]]), tensors.detach().numpy()
        )
        np.testing.assert_array_almost_equal(np.array([2, 2, 3]), lens.detach().numpy())

    def test_label_list_tensors_no_pad_in_vocab(self):
        tensorizers = {
            "intent": LabelListTensorizer(
                label_column="intent", pad_in_vocab=False, allow_unknown=True
            )
        }
        initialize_tensorizers(tensorizers, self.data.train)
        self.assertEqual(8, len(tensorizers["intent"].vocab))
        tensors = []
        for row in self.data.train:
            row["intent"].append("unknown")
            tensors.append(tensorizers["intent"].numberize(row))
        tensors, lens = tensorizers["intent"].tensorize(tensors)
        np.testing.assert_array_almost_equal(
            np.array([[1, 2, 0, -1], [3, 4, 0, -1], [5, 6, 7, 0]]),
            tensors.detach().numpy(),
        )

    def test_label_list_tensors_pad_missing(self):
        ds = SessionPandasDataSource(
            test_df=pd.DataFrame(
                # test None and empty case
                {
                    "session_id": [1, 1, 1, 1],
                    "label": ["positive", "negative", None, ""],
                }
            ),
            schema={"label": List[str]},
            id_col="session_id",
        )
        tensorizers = {
            "label": LabelListTensorizer(
                pad_missing=True,
                label_column="label",
                pad_in_vocab=False,
                allow_unknown=False,
            )
        }
        initialize_tensorizers(tensorizers, ds.test)
        self.assertEqual(2, len(tensorizers["label"].vocab))
        # only one row in test data
        label_idx_list, lens = tensorizers["label"].numberize(next(iter(ds.test)))
        self.assertEqual([0, 1, -1, -1], label_idx_list)

        tensorizers["label"].pad_missing = False
        with self.assertRaises(Exception):
            tensorizers["label"].numberize(next(iter(ds.test)))


# fmt: off
EXPECTED_ACTIONS = [
    [0, 1, 1, 2, 1, 3, 3],
    [4, 1, 5, 1, 1, 1, 3, 1, 6, 7, 8, 1, 3, 1, 3, 3, 1, 1, 9, 1, 1, 1, 1, 1, 3, 1, 3],
    [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [0, 1, 1, 1, 1, 1, 3],
    [11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 7, 1, 3, 3, 1, 5, 12, 1, 3, 3, 1, 1, 1, 9, 1, 1, 1, 3, 1, 3],
    [4, 1, 1, 1, 1, 5, 1, 3, 1, 1, 13, 1, 1, 1, 3, 3],
    [4, 1, 1, 1, 1, 1, 5, 7, 1, 3, 3, 3],
    [14, 1, 1, 1, 6, 1, 3, 1, 5, 1, 3, 3],
    [0, 1, 1, 1, 1, 1, 2, 1, 3, 3],
    [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
]
# fmt: on


class TensorizersTest(unittest.TestCase):
    def setUp(self):
        self.data = TSVDataSource(
            SafeFileWrapper(tests_module.test_file("train_dense_features_tiny.tsv")),
            SafeFileWrapper(tests_module.test_file("test_dense_features_tiny.tsv")),
            eval_file=None,
            field_names=["label", "slots", "text", "dense"],
            schema={"text": str, "label": str},
        )

    def _initialize_tensorizer(self, tensorizer, data=None):
        if data is None:
            data = self.data
        init = tensorizer.initialize()
        init.send(None)  # kick
        for row in data.train:
            init.send(row)
        init.close()

    def test_initialize_tensorizers(self):
        tensorizers = {
            "tokens": TokenTensorizer(text_column="text"),
            "labels": LabelTensorizer(label_column="label"),
            "chars": ByteTensorizer(text_column="text"),
        }
        initialize_tensorizers(tensorizers, self.data.train)
        self.assertEqual(49, len(tensorizers["tokens"].vocab))
        self.assertEqual(7, len(tensorizers["labels"].vocab))

    def test_initialize_token_tensorizer(self):
        # default (build from data)
        tensorizer = TokenTensorizer(text_column="text")
        self._initialize_tensorizer(tensorizer)
        self.assertEqual(49, len(tensorizer.vocab))

        # size limit on tokens from data
        tensorizer = TokenTensorizer(
            text_column="text", vocab_config=VocabConfig(size_from_data=3)
        )
        self._initialize_tensorizer(tensorizer)
        self.assertEqual(5, len(tensorizer.vocab))  # 3 + unk token + pad token

        embed_file = tests_module.test_file("pretrained_embed_raw")

        # vocab from data + vocab_file
        tensorizer = TokenTensorizer(
            text_column="text",
            vocab_config=VocabConfig(
                size_from_data=3,
                vocab_files=[
                    VocabFileConfig(filepath=embed_file, skip_header_line=True)
                ],
            ),
        )
        self._initialize_tensorizer(tensorizer)
        self.assertEqual(15, len(tensorizer.vocab))

        # vocab just from vocab_file
        tensorizer = TokenTensorizer(
            text_column="text",
            vocab_config=VocabConfig(
                build_from_data=False,
                vocab_files=[
                    VocabFileConfig(
                        filepath=embed_file, skip_header_line=True, size_limit=5
                    )
                ],
            ),
        )
        init = tensorizer.initialize()
        # Should skip initialization
        with self.assertRaises(StopIteration):
            init.send(None)
        self.assertEqual(7, len(tensorizer.vocab))  # 5 + unk token + pad token

    def test_numberize_with_token_tensorizer(self):
        tensorizer = TokenTensorizer(text_column="text")
        self._initialize_tensorizer(tensorizer)

        rows = [{"text": "I want some coffee"}, {"text": "Turn it up"}]
        tensors = (tensorizer.numberize(row) for row in rows)
        tokens, seq_len, token_ranges = next(tensors)
        self.assertEqual([24, 0, 0, 0], tokens)
        self.assertEqual(4, seq_len)
        self.assertEqual([(0, 1), (2, 6), (7, 11), (12, 18)], token_ranges)

        tokens, seq_len, token_ranges = next(tensors)
        self.assertEqual([13, 47, 9], tokens)
        self.assertEqual(3, seq_len)
        self.assertEqual([(0, 4), (5, 7), (8, 10)], token_ranges)

    def test_numberize_with_script_token_tensorizer(self):
        rows = [{"text": "I want some coffee"}, {"text": "Turn it up"}]
        expected_outputs = [
            [
                ("tokens", [24, 0, 0, 0]),
                ("seq_len", 4),
                ("token_ranges", [(0, 1), (2, 6), (7, 11), (12, 18)]),
            ],
            [
                ("tokens", [13, 47, 9]),
                ("seq_len", 3),
                ("token_ranges", [(0, 4), (5, 7), (8, 10)]),
            ],
        ]

        tensorizer = ScriptBasedTokenTensorizer(text_column="text")
        self._initialize_tensorizer(tensorizer)
        tensors = (tensorizer.numberize(row) for row in rows)

        for expected, numberized in zip(expected_outputs, tensors):
            for (field, value), out in zip(expected, numberized):
                self.assertEqual(value, out, msg=f"{field} didn't match!")

        # Test that EOS / BOS tokens are correctly added
        tensorizer = ScriptBasedTokenTensorizer(
            text_column="text", add_eos_token=True, add_bos_token=True
        )
        self._initialize_tensorizer(tensorizer)
        tensors = (tensorizer.numberize(row) for row in rows)
        bos = tensorizer.vocab.get_bos_index()
        eos = tensorizer.vocab.get_eos_index()
        unk = tensorizer.vocab.get_unk_index()

        for expected, (tokens, seq_len, token_ranges) in zip(expected_outputs, tensors):
            # Token indices are shifted due to the two new tokens, hence x + 2
            self.assertEqual(
                [bos] + [x + 2 if x != unk else x for x in expected[0][1]] + [eos],
                tokens,
            )

            # We have one extra token at the beginning and one at the end
            self.assertEqual(expected[1][1] + 2, seq_len)

            # EOS / BOS tokens don't have ranges
            self.assertEqual([(-1, -1)] + expected[2][1] + [(-1, -1)], token_ranges)

    def test_tensorize_with_script_token_tensorizer(self):
        rows = [{"text": "I want some coffee"}, {"text": "Turn it up"}]

        tensorizer = ScriptBasedTokenTensorizer(
            text_column="text", add_eos_token=True, add_bos_token=True
        )
        self._initialize_tensorizer(tensorizer)

        def check_results(results):
            tokens, seq_lens, token_ranges = results
            self.assertIsInstance(tokens, torch.LongTensor)
            self.assertIsInstance(seq_lens, torch.LongTensor)
            self.assertIsInstance(token_ranges, torch.LongTensor)

            self.assertEqual((2, 6), tokens.size())
            self.assertEqual((2,), seq_lens.size())
            self.assertEqual((2, 6, 2), token_ranges.size())

            self.assertEqual(
                [[2, 26, 0, 0, 0, 3], [2, 15, 49, 11, 3, 1]], tokens.tolist()
            )
            self.assertEqual([6, 5], seq_lens.tolist())
            self.assertEqual(
                [
                    [[-1, -1], [0, 1], [2, 6], [7, 11], [12, 18], [-1, -1]],
                    [[-1, -1], [0, 4], [5, 7], [8, 10], [-1, -1], [-1, -1]],
                ],
                token_ranges.tolist(),
            )

        original_results = tensorizer.tensorize(
            [tensorizer.numberize(row) for row in rows]
        )
        check_results(original_results)

        # Make sure all is well after exporting to TorchScript
        torchscript_tensorizer = tensorizer.torchscriptify()

        # The tokenizer we use by default is not torchscriptifiable so we do
        # tokenization beforehand and add positions ourselves.
        tokenized_rows = [
            torchscript_tensorizer.tokenize(
                row_text=None, row_pre_tokenized=row["text"].lower().split(" ")
            )
            for row in rows
        ]
        positions_2d = [
            [(-1, -1), (0, 1), (2, 6), (7, 11), (12, 18), (-1, -1)],
            [(-1, -1), (0, 4), (5, 7), (8, 10), (-1, -1)],
        ]

        tokens_2d = []
        seq_lens_1d = []
        for row in tokenized_rows:
            tokens, seq_len, postion = torchscript_tensorizer.numberize(row)
            tokens_2d.append(tokens)
            seq_lens_1d.append(seq_len)

        torchscript_results = torchscript_tensorizer.tensorize(
            tokens_2d, seq_lens_1d, positions_2d
        )
        check_results(torchscript_results)

    def test_create_byte_tensors(self):
        tensorizer = ByteTensorizer(text_column="text", lower=False)
        # not initializing because initializing is a no-op for ByteTensorizer

        s1 = "I want some coffee"
        s2 = "Turn it up"
        s3 = "我不会说中文"
        rows = [{"text": s1}, {"text": s2}, {"text": s3}]
        expected = [list(s1.encode()), list(s2.encode()), list(s3.encode())]

        tensors = [tensorizer.numberize(row) for row in rows]
        self.assertEqual([(bytes, len(bytes)) for bytes in expected], tensors)

    def test_byte_tensors_error_code(self):
        tensorizer = ByteTensorizer(
            text_column="text", lower=False, add_bos_token=True, add_eos_token=True
        )
        s1 = "I want some coffee#"
        s2 = "This is ^the best show I've ever seen"

        rows = [{"text": s1}, {"text": s2}]
        expected_error_code = 1
        with self.assertRaises(SystemExit) as cm:
            for row in rows:
                tensorizer.numberize(row)

        self.assertEqual(cm.exception.code, expected_error_code)

    def test_create_byte_token_tensors(self):
        tensorizer = ByteTokenTensorizer(
            text_column="text", max_seq_len=4, max_byte_len=5
        )
        # not initializing because initializing is a no-op for this tensorizer

        s1 = "I want some coffee today"
        s2 = "Turn it up"

        def ords(word, pad_to):
            return list(word.encode()) + [0] * (pad_to - len(word))

        batch = [{"text": s1}, {"text": s2}]
        # Note that the tokenizer lowercases here
        expected = [
            [ords("i", 5), ords("want", 5), ords("some", 5), ords("coffe", 5)],
            [ords("turn", 5), ords("it", 5), ords("up", 5), ords("", 5)],
        ]
        expected_token_lens = [4, 3]
        expected_byte_lens = [[1, 4, 4, 5], [4, 2, 2, 0]]

        bytes, token_lens, byte_lens = tensorizer.tensorize(
            [tensorizer.numberize(row) for row in batch]
        )
        self.assertIsInstance(bytes, torch.LongTensor)
        self.assertIsInstance(token_lens, torch.LongTensor)
        self.assertIsInstance(byte_lens, torch.LongTensor)
        self.assertEqual((2, 4, 5), bytes.size())
        self.assertEqual((2,), token_lens.size())
        self.assertEqual((2, 4), byte_lens.size())
        self.assertEqual(expected, bytes.tolist())
        self.assertEqual(expected_token_lens, token_lens.tolist())
        self.assertEqual(expected_byte_lens, byte_lens.tolist())

    def test_initialize_label_tensorizer(self):
        tensorizer = LabelTensorizer(label_column="label")
        self._initialize_tensorizer(tensorizer)
        self.assertEqual(7, len(tensorizer.vocab))

    def test_create_label_tensors(self):
        tensorizer = LabelTensorizer(label_column="label")
        self._initialize_tensorizer(tensorizer)

        rows = [
            {"label": "weather/find"},
            {"label": "alarm/set_alarm"},
            {"label": "non/existent"},
        ]

        tensors = (tensorizer.numberize(row) for row in rows)
        tensor = next(tensors)
        self.assertEqual(6, tensor)
        tensor = next(tensors)
        self.assertEqual(1, tensor)
        with self.assertRaises(Exception):
            tensor = next(tensors)

    def test_gazetteer_tensor_bad_json(self):
        tensorizer = GazetteerTensorizer()

        data = TSVDataSource(
            train_file=SafeFileWrapper(
                tests_module.test_file("train_dict_features_bad_json.tsv")
            ),
            test_file=None,
            eval_file=None,
            field_names=["text", "dict"],
            schema={"text": str, "dict": Gazetteer},
        )

        init = tensorizer.initialize()
        init.send(None)  # kick
        with self.assertRaises(Exception):
            for row in data.train:
                init.send(row)
        init.close()

    def test_gazetteer_tensor(self):
        tensorizer = GazetteerTensorizer()

        data = TSVDataSource(
            train_file=SafeFileWrapper(
                tests_module.test_file("train_dict_features.tsv")
            ),
            test_file=None,
            eval_file=None,
            field_names=["text", "dict"],
            schema={"text": str, "dict": Gazetteer},
        )

        self._initialize_tensorizer(tensorizer, data)
        # UNK + PAD + 5 labels
        self.assertEqual(7, len(tensorizer.vocab))

        # only two rows in test file:
        # "Order coffee from Starbucks please"
        # "Order some fries from McDonalds please"
        for i, row in enumerate(data.train):
            if i == 0:
                idx, weights, lens = tensorizer.numberize(row)
                self.assertEqual([1, 1, 2, 3, 1, 1, 4, 1, 1, 1], idx)
                self.assertEqual(
                    [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], weights
                )
                self.assertEqual([1, 2, 1, 1, 1], lens)
            if i == 1:
                idx, weights, lens = tensorizer.numberize(row)
                self.assertEqual([1, 1, 5, 1, 6, 1], idx)
                self.assertEqual([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], weights)
                self.assertEqual([1, 1, 1, 1, 1, 1], lens)

        feats, weights, lens = tensorizer.tensorize(
            tensorizer.numberize(row) for row in data.train
        )
        self.assertEqual(
            [
                [1, 1, 2, 3, 1, 1, 4, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 5, 1, 1, 1, 6, 1, 1, 1],
            ],
            feats.numpy().tolist(),
        )
        self.assertEqual(
            str(
                [
                    [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
            str(
                [[round(w, 2) for w in utt_weights] for utt_weights in weights.numpy()]
            ),
        )
        self.assertEqual(
            [[1, 2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], lens.numpy().tolist()
        )

    def test_seq_tensor(self):
        tensorizer = SeqTokenTensorizer()

        data = TSVDataSource(
            train_file=SafeFileWrapper(
                tests_module.test_file("train_seq_features.tsv")
            ),
            test_file=None,
            eval_file=None,
            field_names=["text_seq"],
            schema={"text_seq": List[str]},
        )

        self._initialize_tensorizer(tensorizer, data)
        # UNK + PAD + 6 tokens
        self.assertEqual(8, len(tensorizer.vocab))

        # only one row in test file:
        # ["where do you wanna meet?", "MPK"]
        for row in data.train:
            tokens, token_lens, seq_lens = tensorizer.prepare_input(row)
            idx, sentence_lens, lens = tensorizer.numberize(row)
            self.assertEqual(2, lens)
            self.assertEqual([[2, 3, 4, 5, 6], [7, 1, 1, 1, 1]], idx)
            self.assertEqual([5, 1], sentence_lens)
            self.assertEqual(2, seq_lens)
            self.assertEqual(
                [
                    ["where", "do", "you", "wanna", "meet?"],
                    [
                        "mpk",
                        str(SpecialTokens.PAD),
                        str(SpecialTokens.PAD),
                        str(SpecialTokens.PAD),
                        str(SpecialTokens.PAD),
                    ],
                ],
                tokens,
            )

    def test_seq_tensor_max_turn(self):
        tensorizer = SeqTokenTensorizer(max_turn=1)

        data = TSVDataSource(
            train_file=SafeFileWrapper(
                tests_module.test_file("train_seq_features.tsv")
            ),
            test_file=None,
            eval_file=None,
            field_names=["text_seq"],
            schema={"text_seq": List[str]},
        )

        self._initialize_tensorizer(tensorizer, data)

        # only one row in test file:
        # ["where do you wanna meet?", "MPK"]
        for row in data.train:
            idx, sentence_lens, seq_len = tensorizer.numberize(row)
            self.assertEqual(1, seq_len)
            self.assertEqual([[2, 3, 4, 5, 6]], idx)
            self.assertEqual([5], sentence_lens)

    def test_seq_tensor_pad_batch(self):
        tensorizer = SeqTokenTensorizer()

        data = TSVDataSource(
            train_file=SafeFileWrapper(
                tests_module.test_file("train_seq_features.tsv")
            ),
            test_file=None,
            eval_file=None,
            field_names=["text_seq"],
            schema={"text_seq": List[str]},
        )

        self._initialize_tensorizer(tensorizer, data)
        token_idx_1 = [[2, 3], [2, 1]]
        token_count_1 = [2, 1]
        seq_len_1 = 2
        token_idx_2 = [[2, 3, 4]]
        token_count_2 = [3]
        seq_len_2 = 1
        token_idx_tensor, token_count_tensor, seq_len_tensor = tensorizer.tensorize(
            [
                (token_idx_1, token_count_1, seq_len_1),
                (token_idx_2, token_count_2, seq_len_2),
            ]
        )
        np.testing.assert_array_almost_equal(
            np.array([[[2, 3, 1], [2, 1, 1]], [[2, 3, 4], [1, 1, 1]]]),
            token_idx_tensor.detach().numpy(),
        )
        np.testing.assert_array_almost_equal(
            np.array([[2, 1], [3, 1]]), token_count_tensor.detach().numpy()
        )
        np.testing.assert_array_almost_equal(
            np.array([2, 1]), seq_len_tensor.detach().numpy()
        )

    def test_seq_tensor_with_bos_eos_eol_bol(self):
        tensorizer = SeqTokenTensorizer(
            add_bos_token=True,
            add_eos_token=True,
            add_bol_token=True,
            add_eol_token=True,
        )

        data = TSVDataSource(
            train_file=SafeFileWrapper(
                tests_module.test_file("train_seq_features.tsv")
            ),
            test_file=None,
            eval_file=None,
            field_names=["text_seq"],
            schema={"text_seq": List[str]},
        )

        self._initialize_tensorizer(tensorizer, data)
        # UNK + PAD + BOS + EOS + BOL + EOL + 6 tokens
        self.assertEqual(12, len(tensorizer.vocab))

        # only one row in test file:
        # ["where do you wanna meet?", "MPK"]
        for row in data.train:
            idx, sen_lens, lens = tensorizer.numberize(row)
            tokens, token_lens, seq_lens = tensorizer.prepare_input(row)
            self.assertEqual(4, lens)
            self.assertEqual(4, seq_lens)
            self.assertEqual([3, 7, 3, 3], token_lens)
            self.assertEqual(
                [
                    [2, 4, 3, 1, 1, 1, 1],
                    [2, 6, 7, 8, 9, 10, 3],
                    [2, 11, 3, 1, 1, 1, 1],
                    [2, 5, 3, 1, 1, 1, 1],
                ],
                idx,
            )
            self.assertEqual(
                [
                    [
                        str(SpecialTokens.BOS),
                        str(SpecialTokens.BOL),
                        str(SpecialTokens.EOS),
                        str(SpecialTokens.PAD),
                        str(SpecialTokens.PAD),
                        str(SpecialTokens.PAD),
                        str(SpecialTokens.PAD),
                    ],
                    [
                        str(SpecialTokens.BOS),
                        "where",
                        "do",
                        "you",
                        "wanna",
                        "meet?",
                        str(SpecialTokens.EOS),
                    ],
                    [
                        str(SpecialTokens.BOS),
                        "mpk",
                        str(SpecialTokens.EOS),
                        str(SpecialTokens.PAD),
                        str(SpecialTokens.PAD),
                        str(SpecialTokens.PAD),
                        str(SpecialTokens.PAD),
                    ],
                    [
                        str(SpecialTokens.BOS),
                        str(SpecialTokens.EOL),
                        str(SpecialTokens.EOS),
                        str(SpecialTokens.PAD),
                        str(SpecialTokens.PAD),
                        str(SpecialTokens.PAD),
                        str(SpecialTokens.PAD),
                    ],
                ],
                tokens,
            )

    def test_create_float_list_tensor(self):
        tensorizer = FloatListTensorizer(
            column="dense", dim=2, error_check=True, normalize=False
        )
        tests = [
            ("[0.1,0.2]", [0.1, 0.2]),  # comma
            ("[0.1, 0.2]", [0.1, 0.2]),  # comma with single space
            ("[0.1,  0.2]", [0.1, 0.2]),  # comma with multiple spaces
            ("[0.1 0.2]", [0.1, 0.2]),  # space
            ("[0.1  0.2]", [0.1, 0.2]),  # multiple spaces
            ("[ 0.1  0.2]", [0.1, 0.2]),  # space after [
            ("[0.1  0.2 ]", [0.1, 0.2]),  # space before ]
            ("[0.  1.]", [0.0, 1.0]),  # 0., 1.
        ]
        for raw, expected in tests:
            row = {"dense": load_float_list(raw)}
            numberized = tensorizer.numberize(row)
            self.assertEqual(expected, numberized)

        precision.FP16_ENABLED = True
        batch = []
        for raw, _ in tests:
            row = {"dense": load_float_list(raw)}
            batch.append(tensorizer.numberize(row))
        tensor = tensorizer.tensorize(batch)
        self.assertEqual(list(tensor.size()), [8, 2])
        self.assertEqual(tensor.dtype, torch.float16)
        precision.FP16_ENABLED = False

    def test_float_list_tensor_prepare_input(self):
        tensorizer = FloatListTensorizer(
            column="dense", dim=2, error_check=True, normalize=False
        )
        tests = [("[0.1,0.2]", [0.1, 0.2])]
        for raw, expected in tests:
            row = {"dense": load_float_list(raw)}
            numberized = tensorizer.prepare_input(row)
            self.assertEqual(expected, numberized)

    def test_create_normalized_float_list_tensor(self):
        def round_list(l):
            return [float("%.4f" % n) for n in l]

        data = TSVDataSource(
            SafeFileWrapper(tests_module.test_file("train_dense_features_tiny.tsv")),
            eval_file=None,
            field_names=["label", "slots", "text", "dense_feat"],
            schema={"text": str, "label": str, "dense_feat": List[float]},
        )
        tensorizer = FloatListTensorizer(
            column="dense_feat", dim=10, error_check=True, normalize=True
        )
        self._initialize_tensorizer(tensorizer, data)
        self.assertEqual(10, tensorizer.normalizer.num_rows)
        self.assertEqual(
            round_list(
                [
                    7.56409,
                    8.2388,
                    0.5531,
                    0.2403,
                    1.03130,
                    6.2888,
                    3.1595,
                    0.1538,
                    0.2403,
                    5.3463,
                ]
            ),
            round_list(tensorizer.normalizer.feature_sums),
        )
        self.assertEqual(
            round_list(
                [
                    5.80172,
                    7.57586,
                    0.30591,
                    0.05774,
                    0.52762,
                    5.22811,
                    2.51727,
                    0.02365,
                    0.05774,
                    4.48798,
                ]
            ),
            round_list(tensorizer.normalizer.feature_squared_sums),
        )
        self.assertEqual(
            round_list(
                [
                    0.75640,
                    0.82388,
                    0.05531,
                    0.02403,
                    0.10313,
                    0.62888,
                    0.31595,
                    0.01538,
                    0.02403,
                    0.53463,
                ]
            ),
            round_list(tensorizer.normalizer.feature_avgs),
        )
        self.assertEqual(
            round_list(
                [
                    0.08953,
                    0.28072,
                    0.16593,
                    0.07209,
                    0.20524,
                    0.35682,
                    0.38974,
                    0.04614,
                    0.07209,
                    0.40369,
                ]
            ),
            round_list(tensorizer.normalizer.feature_stddevs),
        )

        row = [0.64840776, 0.7575, 0.5531, 0.2403, 0, 0.9481, 0, 0.1538, 0.2403, 0.3564]
        output = tensorizer.numberize({"dense_feat": row})

        self.assertEqual(
            round_list(
                [
                    -1.20619,
                    -0.23646,
                    2.99999,
                    3.0,
                    -0.50246,
                    0.89462,
                    -0.81066,
                    2.99999,
                    3.0,
                    -0.44149,
                ]
            ),
            round_list(output),
        )

    def test_create_float_list_seq_tensor(self):
        tensorizer = FloatListSeqTensorizer(column="dense", dim=2, error_check=True)
        tests = [
            (
                ["[0.1,0.2]", "[0.1, 0.2]", "[0.1,  0.2]", "[0.1 0.2]"],
                [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]],
                4,
            ),
            (
                ["[0.1  0.2]", "[ 0.1  0.2]", "[0.1  0.2 ]", "[ 0.1 0.2 ]", "[0.  1.]"],
                [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.0, 1.0]],
                5,
            ),
        ]
        for raw_list, expected, expected_length in tests:
            row = {"dense": [load_float_list(raw) for raw in raw_list]}
            numberized, numberized_len = tensorizer.numberize(row)
            self.assertEqual(expected, numberized)
            self.assertEqual(expected_length, numberized_len)

        batch = []
        for raw_list, _, _ in tests:
            row = {"dense": [load_float_list(raw) for raw in raw_list]}
            tensor, tensor_len = tensorizer.numberize(row)
            batch.append((tensor, tensor_len))

        tensor, tensor_lens = tensorizer.tensorize(batch)
        self.assertEqual(list(tensor.size()), [2, 5, 2])
        self.assertEqual(tensor.dtype, torch.float)
        self.assertEqual(tensor_lens.tolist(), [4, 5])

    def test_float_list_seq_tensor_prepare_input(self):
        tensorizer = FloatListSeqTensorizer(column="dense", dim=2, error_check=True)
        tests = [
            (
                ["[0.1,0.2]", "[0.1, 0.2]", "[0.1,  0.2]", "[0.1 0.2]"],
                [[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]],
                4,
            )
        ]
        for raw_list, expected, expect_length in tests:
            row = {"dense": [load_float_list(raw) for raw in raw_list]}
            numberized, numberized_len = tensorizer.prepare_input(row)
            self.assertEqual(expected, numberized)
            self.assertEqual(expect_length, numberized_len)

    def test_annotation_num(self):
        data = TSVDataSource(
            SafeFileWrapper(tests_module.test_file("compositional_seq2seq_unit.tsv")),
            test_file=None,
            eval_file=None,
            field_names=["text", "seqlogical"],
            schema={"text": str, "seqlogical": str},
        )
        nbrz = AnnotationNumberizer()
        self._initialize_tensorizer(nbrz, data)

        # vocab = {'IN:GET_INFO_TRAFFIC': 0, 'SHIFT': 1, 'SL:LOCATION': 2,
        # 'REDUCE': 3, 'IN:GET_DIRECTIONS': 4, 'SL:DESTINATION': 5, 'SL:SOURCE': 6,
        # 'IN:GET_LOCATION_HOME': 7, 'SL:CONTACT': 8, 'SL:DATE_TIME_DEPARTURE': 9,
        # 'IN:UNSUPPORTED_NAVIGATION': 10, 'IN:GET_ESTIMATED_DURATION': 11,
        # 'IN:GET_LOCATION_WORK': 12, 'SL:PATH_AVOID': 13, 'IN:GET_DISTANCE': 14}

        self.assertEqual(15, len(nbrz.vocab))
        self.assertEqual(1, nbrz.shift_idx)
        self.assertEqual(3, nbrz.reduce_idx)
        self.assertEqual([10], nbrz.ignore_subNTs_roots)
        self.assertEqual(
            [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], nbrz.valid_NT_idxs
        )
        self.assertEqual([0, 4, 7, 10, 11, 12, 14], nbrz.valid_IN_idxs)
        self.assertEqual([2, 5, 6, 8, 9, 13], nbrz.valid_SL_idxs)

        for row, expected in zip(data.train, EXPECTED_ACTIONS):
            actions = nbrz.numberize(row)
            self.assertEqual(expected, actions)


class BERTTensorizerTest(unittest.TestCase):
    def test_bert_tensorizer(self):
        sentence = "<SOS>  Focus Driving School Mulungushi bus station along Kasuba road, wamkopeka building.  Ndola,  Zambia."
        # expected result was obtained offline by running BertModelDataHandler
        expected = [
            101,
            133,
            278,
            217,
            135,
            175,
            287,
            766,
            462,
            100,
            379,
            182,
            459,
            334,
            459,
            280,
            504,
            462,
            425,
            283,
            171,
            462,
            567,
            474,
            180,
            262,
            217,
            459,
            931,
            262,
            913,
            117,
            192,
            262,
            407,
            478,
            287,
            744,
            263,
            478,
            262,
            560,
            119,
            183,
            282,
            287,
            843,
            117,
            195,
            262,
            407,
            931,
            566,
            119,
            102,
        ]
        row = {"text": sentence}
        tensorizer = BERTTensorizer.from_config(
            BERTTensorizer.Config(
                tokenizer=WordPieceTokenizer.Config(
                    wordpiece_vocab_path="pytext/data/test/data/wordpiece_1k.txt"
                )
            )
        )
        tensorizer_impl = BERTTensorizerScriptImpl(
            tokenizer=DoNothingTokenizer(),
            vocab=tensorizer.vocab,
            max_seq_len=tensorizer.max_seq_len,
        ).torchscriptify()

        tokens, segment_label, seq_len, positions = tensorizer.numberize(row)
        self.assertEqual(tokens, expected)
        self.assertEqual(seq_len, len(expected))
        self.assertEqual(segment_label, [0] * len(expected))

        tokens, pad_mask, segment_labels, _ = tensorizer.tensorize(
            [(tokens, segment_label, seq_len, positions)]
        )
        self.assertEqual(pad_mask[0].tolist(), [1] * len(expected))

        per_sentence_tokens = [tensorizer.tokenizer.tokenize(sentence)]
        tokens, segment_label, seq_len, positions = tensorizer_impl.numberize(
            per_sentence_tokens
        )
        self.assertEqual(tokens, expected)
        self.assertEqual(seq_len, len(expected))
        self.assertEqual(segment_label, [0] * len(expected))

        tokens, pad_mask, segment_labels, _ = tensorizer_impl.tensorize(
            [tokens], [segment_label], [seq_len], [positions]
        )
        self.assertEqual(pad_mask[0].tolist(), [1] * len(expected))

    def test_bert_pair_tensorizer(self):
        sentences = ["Focus", "Driving School"]
        expected_tokens = [101, 175, 287, 766, 462, 102, 100, 379, 102]
        expected_segment_labels = [0, 0, 0, 0, 0, 0, 1, 1, 1]
        row = {"text1": sentences[0], "text2": sentences[1]}
        tensorizer = BERTTensorizer.from_config(
            BERTTensorizer.Config(
                columns=["text1", "text2"],
                tokenizer=WordPieceTokenizer.Config(
                    wordpiece_vocab_path="pytext/data/test/data/wordpiece_1k.txt"
                ),
            )
        )
        tokens, segment_labels, seq_len, _ = tensorizer.numberize(row)
        self.assertEqual(tokens, expected_tokens)
        self.assertEqual(segment_labels, expected_segment_labels)
        self.assertEqual(seq_len, len(expected_tokens))


class RobertaTensorizerTest(unittest.TestCase):
    def test_roberta_tensorizer(self):
        text = "Prototype"
        tokens = [[0, 4, 5, 2]]
        pad_masks = [[1, 1, 1, 1]]
        segment_labels = [[0, 0, 0, 0]]
        positions = [[0, 1, 2, 3]]
        expected = [tokens, pad_masks, segment_labels, positions]

        tensorizer = RoBERTaTensorizer.from_config(
            RoBERTaTensorizer.Config(
                tokenizer=GPT2BPETokenizer.Config(
                    bpe_encoder_path="pytext/data/test/data/gpt2_encoder.json",
                    bpe_vocab_path="pytext/data/test/data/gpt2_vocab.bpe",
                ),
                vocab_file="pytext/data/test/data/gpt2_dict.txt",
                max_seq_len=256,
            )
        )
        tensors = tensorizer.tensorize([tensorizer.numberize({"text": text})])
        for tensor, expect in zip(tensors, expected):
            self.assertEqual(tensor.tolist(), expect)

        tensorizer_impl = RoBERTaTensorizerScriptImpl(
            tokenizer=DoNothingTokenizer(),
            vocab=tensorizer.vocab,
            max_seq_len=tensorizer.max_seq_len,
        )
        script_tensorizer_impl = tensorizer_impl.torchscriptify()
        per_sentence_tokens = [tensorizer.tokenizer.tokenize(text)]
        tokens_2d, segment_labels_2d, seq_lens_1d, positions_2d = zip(
            *[script_tensorizer_impl.numberize(per_sentence_tokens)]
        )
        script_tensors = script_tensorizer_impl.tensorize(
            tokens_2d, segment_labels_2d, seq_lens_1d, positions_2d
        )
        for tensor, expect in zip(script_tensors, expected):
            self.assertEqual(tensor.tolist(), expect)
        # test it is able to call torchscriptify multiple time
        tensorizer_impl.torchscriptify()


class SquadForRobertaTensorizerTest(unittest.TestCase):
    def test_squad_roberta_tensorizer(self):
        row = {
            "id": 0,
            "doc": "Prototype",
            "question": "otype",
            "answers": ["Prot"],
            "answer_starts": [0],
            "has_answer": True,
        }
        tensorizer = SquadForRoBERTaTensorizer.from_config(
            SquadForRoBERTaTensorizer.Config(
                tokenizer=GPT2BPETokenizer.Config(
                    bpe_encoder_path="pytext/data/test/data/gpt2_encoder.json",
                    bpe_vocab_path="pytext/data/test/data/gpt2_vocab.bpe",
                ),
                vocab_file="pytext/data/test/data/gpt2_dict.txt",
                max_seq_len=250,
            )
        )
        tokens, segments, seq_len, positions, start, end = tensorizer.numberize(row)
        # check against manually verified answer positions in tokenized output
        # there are 4 identical answers
        self.assertEqual(start, [3])
        self.assertEqual(end, [3])
        self.assertEqual(len(tokens), seq_len)
        self.assertEqual(len(segments), seq_len)


class SquadForBERTTensorizerTest(unittest.TestCase):
    def test_squad_tensorizer(self):
        source = SquadDataSource.from_config(
            SquadDataSource.Config(
                eval_filename=tests_module.test_file("squad_tiny.json")
            )
        )
        row = next(iter(source.eval))
        tensorizer = SquadForBERTTensorizer.from_config(
            SquadForBERTTensorizer.Config(
                tokenizer=WordPieceTokenizer.Config(
                    wordpiece_vocab_path="pytext/data/test/data/wordpiece_1k.txt"
                ),
                max_seq_len=250,
            )
        )
        tokens, segments, seq_len, positions, start, end = tensorizer.numberize(row)
        # check against manually verified answer positions in tokenized output
        # there are 4 identical answers
        self.assertEqual(start, [83, 83, 83, 83])
        self.assertEqual(end, [87, 87, 87, 87])
        self.assertEqual(len(tokens), seq_len)
        self.assertEqual(len(segments), seq_len)

        tensorizer.max_seq_len = 50
        # answer should be truncated out
        _, _, _, _, start, end = tensorizer.numberize(row)
        self.assertEqual(start, [-100, -100, -100, -100])
        self.assertEqual(end, [-100, -100, -100, -100])
        self.assertEqual(len(tokens), seq_len)
        self.assertEqual(len(segments), seq_len)


class SquadTensorizerTest(unittest.TestCase):
    def setUp(self):
        self.json_data_source = SquadDataSource.from_config(
            SquadDataSource.Config(
                train_filename=tests_module.test_file("squad_tiny.json"),
                eval_filename=None,
                test_filename=None,
            )
        )
        self.tsv_data_source = SquadDataSource.from_config(
            SquadDataSource.Config(
                train_filename=tests_module.test_file("squad_tiny.tsv"),
                eval_filename=None,
                test_filename=None,
            )
        )

        self.tensorizer_with_wordpiece = SquadTensorizer.from_config(
            SquadTensorizer.Config(
                tokenizer=WordPieceTokenizer.Config(
                    wordpiece_vocab_path="pytext/data/test/data/wordpiece_1k.txt"
                ),
                max_seq_len=250,
            )
        )
        self.tensorizer_with_alphanumeric = SquadTensorizer.from_config(
            SquadTensorizer.Config(
                tokenizer=Tokenizer.Config(split_regex=r"\W+"), max_seq_len=250
            )
        )

    def _init_tensorizer(self, tsv=False):
        tensorizer_dict = {
            "wordpiece": self.tensorizer_with_wordpiece,
            "alphanumeric": self.tensorizer_with_alphanumeric,
        }
        data_source = self.tsv_data_source.train if tsv else self.json_data_source.train
        initialize_tensorizers(tensorizer_dict, data_source)

    def test_initialize(self):
        self._init_tensorizer()
        self.assertEqual(len(self.tensorizer_with_wordpiece.vocab), 1000)
        self.assertEqual(
            len(self.tensorizer_with_wordpiece.ques_tensorizer.vocab), 1000
        )
        self.assertEqual(len(self.tensorizer_with_wordpiece.doc_tensorizer.vocab), 1000)
        self.assertEqual(len(self.tensorizer_with_alphanumeric.vocab), 1418)
        self.assertEqual(
            len(self.tensorizer_with_alphanumeric.ques_tensorizer.vocab), 1418
        )
        self.assertEqual(
            len(self.tensorizer_with_alphanumeric.doc_tensorizer.vocab), 1418
        )

    def test_numberize_with_alphanumeric(self):
        self._init_tensorizer()
        row = next(iter(self.json_data_source.train))
        (
            doc_tokens,
            doc_seq_len,
            ques_tokens,
            ques_seq_len,
            answer_start_token_idx,
            answer_end_token_idx,
        ) = self.tensorizer_with_alphanumeric.numberize(row)

        # check against manually verified answer positions in tokenized output
        # there are 4 identical answers
        self.assertEqual(len(ques_tokens), ques_seq_len)
        self.assertEqual(len(doc_tokens), doc_seq_len)
        self.assertEqual(ques_tokens, [2, 3, 4, 5, 6, 7])  # It's a coincidence.
        self.assertEqual(answer_start_token_idx, [26, 26, 26, 26])
        self.assertEqual(answer_end_token_idx, [26, 26, 26, 26])

        self.tensorizer_with_alphanumeric.doc_tensorizer.max_seq_len = 20
        # answer should be truncated out because max doc len is smaller.
        (
            doc_tokens,
            doc_seq_len,
            ques_tokens,
            ques_seq_len,
            answer_start_token_idx,
            answer_end_token_idx,
        ) = self.tensorizer_with_alphanumeric.numberize(row)
        self.assertEqual(len(ques_tokens), ques_seq_len)
        self.assertEqual(len(doc_tokens), doc_seq_len)
        self.assertEqual(answer_start_token_idx, [-100])
        self.assertEqual(answer_end_token_idx, [-100])

    def test_numberize_with_wordpiece(self):
        self._init_tensorizer()
        row = next(iter(self.json_data_source.train))
        (
            doc_tokens,
            doc_seq_len,
            ques_tokens,
            ques_seq_len,
            answer_start_token_idx,
            answer_end_token_idx,
        ) = self.tensorizer_with_wordpiece.numberize(row)

        # check against manually verified answer positions in tokenized output
        # there are 4 identical answers
        self.assertEqual(len(ques_tokens), ques_seq_len)
        self.assertEqual(len(doc_tokens), doc_seq_len)
        self.assertEqual(answer_start_token_idx, [70, 70, 70, 70])
        self.assertEqual(answer_end_token_idx, [74, 74, 74, 74])

        self.tensorizer_with_wordpiece.doc_tensorizer.max_seq_len = 50
        # answer should be truncated out because max doc len is smaller.
        (
            doc_tokens,
            doc_seq_len,
            ques_tokens,
            ques_seq_len,
            answer_start_token_idx,
            answer_end_token_idx,
        ) = self.tensorizer_with_wordpiece.numberize(row)
        self.assertEqual(len(ques_tokens), ques_seq_len)
        self.assertEqual(len(doc_tokens), doc_seq_len)
        self.assertEqual(answer_start_token_idx, [-100])
        self.assertEqual(answer_end_token_idx, [-100])

    def test_tsv_numberize_with_alphanumeric(self):
        # No need to repeat other tests with TSV.
        # All we want to test is that TSV and JSON loading are identical.
        self._init_tensorizer(tsv=True)
        row = next(iter(self.json_data_source.train))
        print(row)
        (
            doc_tokens,
            doc_seq_len,
            ques_tokens,
            ques_seq_len,
            answer_start_token_idx,
            answer_end_token_idx,
        ) = self.tensorizer_with_alphanumeric.numberize(row)

        # check against manually verified answer positions in tokenized output
        # there are 4 identical answers
        self.assertEqual(len(ques_tokens), ques_seq_len)
        self.assertEqual(len(doc_tokens), doc_seq_len)
        self.assertEqual(ques_tokens, [2, 3, 4, 5, 6, 7])  # It's a coincidence.
        self.assertEqual(answer_start_token_idx, [26, 26, 26, 26])
        self.assertEqual(answer_end_token_idx, [26, 26, 26, 26])

        self.tensorizer_with_alphanumeric.doc_tensorizer.max_seq_len = 20
        # answer should be truncated out because max doc len is smaller.
        (
            doc_tokens,
            doc_seq_len,
            ques_tokens,
            ques_seq_len,
            answer_start_token_idx,
            answer_end_token_idx,
        ) = self.tensorizer_with_alphanumeric.numberize(row)
        self.assertEqual(len(ques_tokens), ques_seq_len)
        self.assertEqual(len(doc_tokens), doc_seq_len)
        self.assertEqual(answer_start_token_idx, [-100])
        self.assertEqual(answer_end_token_idx, [-100])
