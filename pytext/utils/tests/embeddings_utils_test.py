#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import tempfile
import unittest

import numpy as np
from pytext.common.constants import DatasetFieldName
from pytext.config.field_config import (
    DocLabelConfig,
    EmbedInitStrategy,
    FeatureConfig,
    WordFeatConfig,
    WordLabelConfig,
)
from pytext.data.featurizer import SimpleFeaturizer
from pytext.utils.embeddings import PretrainedEmbedding
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()
TRAIN_FILE = tests_module.test_file("train_data_tiny.tsv")
EVAL_FILE = tests_module.test_file("test_data_tiny.tsv")
TEST_FILE = tests_module.test_file("test_data_tiny.tsv")


VOCAB_SIZE = 10
EMBED_DIM = 5
EMBED_RAW_PATH = tests_module.test_file("pretrained_embed_raw")
EMBED_CACHED_PATH = tests_module.test_file("test_embed.cached")
EMBED_XLU_CACHED_PATH = tests_module.test_file("test_embed_xlu.cached")


class PretrainedEmbedsTest(unittest.TestCase):
    def test_load_pretrained_embeddings(self):
        pretrained_emb = PretrainedEmbedding(EMBED_RAW_PATH)

        self.assertEqual(len(pretrained_emb.embed_vocab), VOCAB_SIZE)
        self.assertEqual(pretrained_emb.embed_vocab[0], "</s>")
        self.assertEqual(pretrained_emb.embed_vocab[2], "to")

        self.assertEqual(len(pretrained_emb.stoi), VOCAB_SIZE)
        self.assertEqual(pretrained_emb.stoi["</s>"], 0)
        self.assertEqual(pretrained_emb.stoi["to"], 2)

        self.assertEqual(pretrained_emb.embedding_vectors.size(0), VOCAB_SIZE)
        self.assertEqual(pretrained_emb.embedding_vectors.size(1), EMBED_DIM)

    def test_cache_embeds(self):
        embeddings_ref = PretrainedEmbedding()
        embeddings_ref.load_pretrained_embeddings(EMBED_RAW_PATH)
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".{}".format("cached")
        ) as cached_path:
            embeddings_ref.cache_pretrained_embeddings(cached_path.name)
            embeddings_cached = PretrainedEmbedding()
            embeddings_cached.load_cached_embeddings(cached_path.name)

        np.testing.assert_array_equal(
            sorted(embeddings_cached.stoi.keys()), sorted(embeddings_ref.stoi.keys())
        )
        np.testing.assert_array_equal(
            embeddings_cached.embed_vocab, embeddings_ref.embed_vocab
        )
        np.testing.assert_array_equal(
            sorted(embeddings_cached.stoi.values()),
            sorted(embeddings_ref.stoi.values()),
        )
        for word_idx in embeddings_ref.stoi.values():
            np.testing.assert_array_almost_equal(
                embeddings_cached.embedding_vectors[word_idx],
                embeddings_ref.embedding_vectors[word_idx],
            )

    def test_assign_pretrained_weights(self):
        embeddings_ref = PretrainedEmbedding()
        embeddings_ref.load_cached_embeddings(EMBED_CACHED_PATH)
        VOCAB = ["UNK", "aloha", "the"]
        embed_vocab_to_idx = {tok: i for i, tok in enumerate(VOCAB)}
        pretrained_embeds = embeddings_ref.initialize_embeddings_weights(
            embed_vocab_to_idx, "UNK", EMBED_DIM, EmbedInitStrategy.RANDOM
        )
        assert pretrained_embeds.shape[0] == len(VOCAB)
        assert pretrained_embeds.shape[1] == EMBED_DIM
        np.testing.assert_array_almost_equal(
            pretrained_embeds[1].numpy(),
            [-0.43124, 0.014934, -0.50635, 0.60506, 0.56051],
        )  # embedding vector for 'aloha'
        np.testing.assert_array_almost_equal(
            pretrained_embeds[2].numpy(),
            [-0.39153, -0.19803, 0.2573, -0.18617, 0.25551],
        )  # embedding vector for 'the'

    def test_cache_xlu_embeds(self):
        embeddings_ref = PretrainedEmbedding()

        dialects = ["en_US", "en_UK", "es_XX"]
        for dialect in dialects:
            embeddings_ref.load_pretrained_embeddings(
                EMBED_RAW_PATH, append=True, dialect=dialect
            )
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".{}".format("cached")
        ) as cached_path:
            embeddings_ref.cache_pretrained_embeddings(cached_path.name)
            embeddings_cached = PretrainedEmbedding()
            embeddings_cached.load_cached_embeddings(cached_path.name)

        np.testing.assert_array_equal(
            sorted(embeddings_cached.stoi.keys()), sorted(embeddings_ref.stoi.keys())
        )
        np.testing.assert_array_equal(
            embeddings_cached.embed_vocab, embeddings_ref.embed_vocab
        )
        np.testing.assert_array_equal(
            sorted(embeddings_cached.stoi.values()),
            sorted(embeddings_ref.stoi.values()),
        )
        for word_idx in embeddings_ref.stoi.values():
            np.testing.assert_array_almost_equal(
                embeddings_cached.embedding_vectors[word_idx],
                embeddings_ref.embedding_vectors[word_idx],
            )

    def test_assign_pretrained_xlu_weights(self):
        embeddings_ref = PretrainedEmbedding()
        embeddings_ref.load_cached_embeddings(EMBED_XLU_CACHED_PATH)
        VOCAB = ["UNK", "aloha-en_US", "the-es_XX"]
        embed_vocab_to_idx = {tok: i for i, tok in enumerate(VOCAB)}
        pretrained_embeds = embeddings_ref.initialize_embeddings_weights(
            embed_vocab_to_idx, "UNK", EMBED_DIM, EmbedInitStrategy.RANDOM
        )
        assert pretrained_embeds.shape[0] == len(VOCAB)
        assert pretrained_embeds.shape[1] == EMBED_DIM
        np.testing.assert_array_almost_equal(
            pretrained_embeds[1].numpy(),
            [-0.43124, 0.014934, -0.50635, 0.60506, 0.56051],
        )  # embedding vector for 'aloha-en_US'
        np.testing.assert_array_almost_equal(
            pretrained_embeds[2].numpy(),
            [-0.39153, -0.19803, 0.2573, -0.18617, 0.25551],
        )  # embedding vector for 'the-es_XX'
