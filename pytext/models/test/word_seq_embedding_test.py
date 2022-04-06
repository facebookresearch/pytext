#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import tempfile
import unittest

import torch
from caffe2.python.onnx import backend as caffe2_backend
from pytext.models.embeddings.word_seq_embedding import WordSeqEmbedding
from pytext.models.representations.bilstm import BiLSTM
from torch.onnx import ExportTypes, OperatorExportTypes


class WordSeqEmbeddingTest(unittest.TestCase):
    def test_basic(self):
        # Setup embedding
        num_embeddings = 5
        lstm_dim = 8
        embedding_module = WordSeqEmbedding(
            lstm_config=BiLSTM.Config(
                lstm_dim=lstm_dim, num_layers=2, bidirectional=True
            ),
            num_embeddings=num_embeddings,
            word_embed_dim=4,
            embeddings_weight=None,
            init_range=[-1, 1],
            unk_token_idx=4,
        )
        # bidirectional
        output_dim = lstm_dim * 2
        self.assertEqual(embedding_module.embedding_dim, output_dim)

        # Check output shape
        input_batch_size, max_seq_len, max_token_count = 4, 3, 5
        token_seq_idx = torch.randint(
            low=0,
            high=num_embeddings,
            size=[input_batch_size, max_seq_len, max_token_count],
        )
        seq_token_count = torch.randint(
            low=1, high=max_token_count, size=[input_batch_size, max_seq_len]
        )
        output_embedding = embedding_module(token_seq_idx, seq_token_count)

        expected_output_dims = [input_batch_size, max_seq_len, output_dim]
        self.assertEqual(list(output_embedding.size()), expected_output_dims)

    def test_onnx_export(self):
        # Setup embedding
        num_embeddings = 5
        lstm_dim = 8
        embedding_module = WordSeqEmbedding(
            lstm_config=BiLSTM.Config(
                lstm_dim=lstm_dim, num_layers=2, bidirectional=True
            ),
            num_embeddings=num_embeddings,
            word_embed_dim=4,
            embeddings_weight=None,
            init_range=[-1, 1],
            unk_token_idx=4,
        )
        input_batch_size, max_seq_len, max_token_count = 1, 3, 5
        seq_token_idx = torch.randint(
            low=0,
            high=num_embeddings,
            size=[input_batch_size, max_seq_len, max_token_count],
        )
        seq_token_count = torch.randint(
            low=1, high=max_token_count, size=[input_batch_size, max_seq_len]
        )
        dummy_inputs = (seq_token_idx, seq_token_count)
        with tempfile.TemporaryFile() as tmp_file:
            with torch.no_grad():
                torch.onnx._export(
                    embedding_module,
                    dummy_inputs,
                    tmp_file,
                    input_names=["seq_token_idx", "seq_token_count"],
                    output_names=["embedding"],
                    export_params=True,
                    operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
                    opset_version=9,
                    export_type=ExportTypes.ZIP_ARCHIVE,
                )
            # make sure caffe2 can load
            caffe2_backend.prepare_zip_archive(tmp_file)
