#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.data import utils


class TargetTest(unittest.TestCase):
    def test_align_target_label(self):
        target = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        batch_label_list = [["l3", "l2", "l1"], ["l1", "l3", "l2"]]
        label_vocab = {"l1": 0, "l2": 1, "l3": 2}
        align_target = utils.align_target_labels(target, batch_label_list, label_vocab)
        self.assertListEqual(align_target, [[0.3, 0.2, 0.1], [0.1, 0.3, 0.2]])


class PaddingTest(unittest.TestCase):
    def testPadding(self):
        self.assertEqual(
            [[1, 2, 3], [1, 0, 0]], utils.pad([[1, 2, 3], [1]], pad_token=0)
        )
        self.assertEqual(
            [[[1], [2], [3]], [[1], [0], [0]]],
            utils.pad([[[1], [2], [3]], [[1]]], pad_token=0),
        )
        self.assertEqual(
            [[1, 2, 3, 4, 5, 6, 7], [9, 9, 9, 9, 9, 9, 9]],
            utils.pad([[1, 2, 3, 4, 5, 6, 7], []], pad_token=9),
        )

    def testPaddingProvideShape(self):
        self.assertEqual(
            [[0, 0, 0], [0, 0, 0]], utils.pad([], pad_token=0, pad_shape=(2, 3))
        )
        self.assertEqual(
            [[1, 2, 3], [1, 0, 0]],
            utils.pad([[1, 2, 3], [1]], pad_token=0, pad_shape=(2, 3)),
        )
        self.assertEqual([], utils.pad([], pad_token=0, pad_shape=()))


class VocabularyTest(unittest.TestCase):
    def testBuildVocabulary(self):
        tokens = """
            your bones don't break mine do that's clear your cells react to
            bacteria and viruses differently than mine you don't get sick
            i do that's also clear but for some reason you and i react the
            exact same way to water we swallow it too fast we choke we get
            some in our lungs we drown however unreal it may seem we are
            connected you and i we're on the same curve just on opposite ends
        """.split()
        builder = utils.VocabBuilder()
        builder.add_all(tokens)
        vocab = builder.make_vocab()
        self.assertEqual(54, len(vocab))

        indices = vocab.lookup_all(["can i get a coffee".split()])
        self.assertEqual([[0, 21, 19, 0, 0]], indices)
        indices = vocab.lookup_all_internal(
            ["your unk unk unk unk unk unk unk unk unk".split()]
        )
        self.assertEqual(0.9, indices[1] / indices[2])
        indices = vocab.lookup_all_internal(
            [["bones unk unk unk unk unk".split()], ["bones on on on".split()]]
        )
        self.assertEqual(0.5, indices[1] / indices[2])
        indices = vocab.lookup_all_internal(
            [["bones unk unk unk unk".split()], ["unk unk unk unk unk".split()]]
        )
        self.assertEqual(0.9, indices[1] / indices[2])
