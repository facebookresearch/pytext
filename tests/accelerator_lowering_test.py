#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import torch
from pytext.config import ExportConfig
from pytext.task.accelerator_lowering import accelerator_transformerLayers_inputs


class TestAcceleratorLowering(unittest.TestCase):
    def testEmptySeqPaddingConfigThrowsException(self):
        empty_export_config = ExportConfig(batch_padding_control=[0, 5, 8])
        model = DummyModel(max_seq_len=22, embedding_dim=10)
        script_func = torch.jit.script(model)
        with self.assertRaises(RuntimeError):
            accelerator_transformerLayers_inputs(
                model, script_func, empty_export_config, None, ""
            )

    def testEmptyBatchPaddingConfigThrowsException(self):
        empty_export_config = ExportConfig(seq_padding_control=[0, 10, 20])
        model = DummyModel(max_seq_len=22, embedding_dim=10)
        script_func = torch.jit.script(model)
        with self.assertRaises(RuntimeError):
            accelerator_transformerLayers_inputs(
                model, script_func, empty_export_config, None, ""
            )

    def testSeqPaddingLimitedBymaxSeqLen(self):
        model = DummyModel(max_seq_len=10, embedding_dim=32)
        script_func = torch.jit.script(model)
        export_config = ExportConfig(
            seq_padding_control=[0, 5, 50], batch_padding_control=[0, 15]
        )
        input_examples = accelerator_transformerLayers_inputs(
            model, script_func, export_config, None, ""
        )

        # effective seq padding [5, 10]
        self.assertEqual(len(input_examples), 2)

    def testNonPositiveSeqPaddingIgnored(self):
        model = DummyModel(max_seq_len=10, embedding_dim=32)
        script_func = torch.jit.script(model)
        export_config = ExportConfig(
            seq_padding_control=[-2, 0], batch_padding_control=[0, 15]
        )
        input_examples = accelerator_transformerLayers_inputs(
            model, script_func, export_config, None, ""
        )

        # only default max_seq_length used for seq padding
        self.assertEqual(len(input_examples), 1)

    def testNonPositiveBatchPaddingIgnored(self):
        model = DummyModel(max_seq_len=10, embedding_dim=32)
        script_func = torch.jit.script(model)
        export_config = ExportConfig(
            seq_padding_control=[22], batch_padding_control=[0]
        )
        input_examples = accelerator_transformerLayers_inputs(
            model, script_func, export_config, None, ""
        )

        self.assertEqual(len(input_examples), 0)

    def testReturnWithCorrectShape(self):
        model = DummyModel(max_seq_len=10, embedding_dim=32)
        script_func = torch.jit.script(model)
        export_config = ExportConfig(
            seq_padding_control=[0, 5], batch_padding_control=[0, 15]
        )
        input_examples = accelerator_transformerLayers_inputs(
            model, script_func, export_config, None, ""
        )
        self.assertEqual(len(input_examples), 2)
        self.assertEqual(input_examples[0][0].get_dims(), [5, 15, 32])
        self.assertEqual(input_examples[0][1].get_dims(), [15, 5])
        self.assertEqual(input_examples[1][0].get_dims(), [10, 15, 32])
        self.assertEqual(input_examples[1][1].get_dims(), [15, 10])

    # TODO: When iterator argument is actually used to generated examples
    def testIterator(self):
        pass


class DummyModel(torch.nn.Module):
    class TestTokenEmbedding:
        def __init__(self, embedding_dim):
            self.embedding_dim = embedding_dim

    def __init__(self, max_seq_len, embedding_dim):
        super(DummyModel, self).__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = self.TestTokenEmbedding(embedding_dim=embedding_dim)

    @torch.jit.export
    def get_max_seq_len(self):
        return self.max_seq_len

    def forward(self):
        pass
