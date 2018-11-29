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
from pytext.data import JointModelDataHandler
from pytext.data.featurizer import SimpleFeaturizer
from pytext.utils.embeddings_utils import PretrainedEmbedding
from pytext.utils.test_utils import tests_module


TRAIN_FILE = tests_module.test_file("train_data_tiny.tsv")
EVAL_FILE = tests_module.test_file("test_data_tiny.tsv")
TEST_FILE = tests_module.test_file("test_data_tiny.tsv")


EMBED_RAW_PATH = tests_module.test_file("pretrained_embed_raw")
EMBED_CACHED_PATH = tests_module.test_file("test_embed.cached")
EMBED_XLU_CACHED_PATH = tests_module.test_file("test_embed_xlu.cached")


class PretrainedEmbedsTest(unittest.TestCase):
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

    def test_assing_pretrained_weights(self):
        embeddings_ref = PretrainedEmbedding()
        embeddings_ref.load_cached_embeddings(EMBED_CACHED_PATH)
        VOCAB = ["UNK", "aloha", "the"]
        EMBED_DIM = 5
        # Get Vocab to Idx:
        UNK_IDX = 0
        embed_vocab_to_idx = {}
        for word in embeddings_ref.embed_vocab:
            if word in VOCAB:
                embed_vocab_to_idx[word] = VOCAB.index(word)
            else:
                embed_vocab_to_idx[word] = UNK_IDX
        pretrained_embeds = embeddings_ref.initialize_embeddings_weights(
            embed_vocab_to_idx, UNK_IDX, len(VOCAB), EMBED_DIM, EmbedInitStrategy.RANDOM
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

    def test_assing_pretrained_xlu_weights(self):
        embeddings_ref = PretrainedEmbedding()
        embeddings_ref.load_cached_embeddings(EMBED_XLU_CACHED_PATH)
        VOCAB = ["UNK", "aloha-en_US", "the-es_XX"]
        EMBED_DIM = 5
        # Get Vocab to Idx:
        UNK_IDX = 0
        embed_vocab_to_idx = {}
        for word in embeddings_ref.embed_vocab:
            if word in VOCAB:
                embed_vocab_to_idx[word] = VOCAB.index(word)
            else:
                embed_vocab_to_idx[word] = UNK_IDX
        pretrained_embeds = embeddings_ref.initialize_embeddings_weights(
            embed_vocab_to_idx, UNK_IDX, len(VOCAB), EMBED_DIM, EmbedInitStrategy.RANDOM
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

    def test_intializing_embeds_from_config(self):
        feature_config = FeatureConfig(
            word_feat=WordFeatConfig(
                embedding_init_strategy=EmbedInitStrategy.RANDOM,
                embed_dim=5,
                pretrained_embeddings_path=tests_module.TEST_BASE_DIR,
            )
        )
        data_handler = JointModelDataHandler.from_config(
            JointModelDataHandler.Config(),
            feature_config,
            [DocLabelConfig(), WordLabelConfig()],
            featurizer=SimpleFeaturizer.from_config(
                SimpleFeaturizer.Config(), feature_config
            ),
        )

        data_handler.init_metadata_from_path(TRAIN_FILE, EVAL_FILE, TEST_FILE)

        pretrained_embeds = data_handler.metadata.features[
            DatasetFieldName.TEXT_FIELD
        ].pretrained_embeds_weight
        # test random initialization (values should be non-0)
        np.testing.assert_array_less(
            [0, 0, 0, 0, 0], np.absolute(pretrained_embeds[11].numpy())
        )

        feature_config = FeatureConfig(
            word_feat=WordFeatConfig(
                embedding_init_strategy=EmbedInitStrategy.ZERO,
                embed_dim=5,
                pretrained_embeddings_path=tests_module.TEST_BASE_DIR,
            )
        )
        data_handler = JointModelDataHandler.from_config(
            JointModelDataHandler.Config(),
            feature_config,
            [DocLabelConfig(), WordLabelConfig()],
            featurizer=SimpleFeaturizer.from_config(
                SimpleFeaturizer.Config(), feature_config
            ),
        )
        data_handler.init_metadata_from_path(TRAIN_FILE, EVAL_FILE, TEST_FILE)

        pretrained_embeds = data_handler.metadata.features[
            DatasetFieldName.TEXT_FIELD
        ].pretrained_embeds_weight
        # test zero initialization (values should all be 0)
        np.testing.assert_array_equal([0, 0, 0, 0, 0], pretrained_embeds[11].numpy())
