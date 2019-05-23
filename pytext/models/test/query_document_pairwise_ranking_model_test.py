#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

import numpy as np
from pytext.config.field_config import FeatureConfig, WordFeatConfig
from pytext.config.query_document_pairwise_ranking import ModelInputConfig
from pytext.data import QueryDocumentPairwiseRankingDataHandler
from pytext.data.featurizer import SimpleFeaturizer
from pytext.models.decoders.mlp_decoder_query_response import MLPDecoderQueryResponse
from pytext.models.output_layers import PairwiseRankingOutputLayer
from pytext.models.query_document_pairwise_ranking_model import (
    QueryDocumentPairwiseRankingModel_Deprecated,
)
from pytext.models.representations.query_document_pairwise_ranking_rep import (
    QueryDocumentPairwiseRankingRep,
)
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


def to_numpy(a_tensor):
    return a_tensor.detach().numpy()


class QueryDocumentPairwiseRankingModelTest(unittest.TestCase):
    def setUp(self):
        self.setup_data()
        self.model_shared_rep = self.create_model(True)
        self.model_unshared_rep = self.create_model(False)

    def setup_data(self):
        simple_featurizer_config = SimpleFeaturizer.Config()
        simple_featurizer_config.split_regex = r""
        simple_featurizer_config.convert_to_bytes = True

        self.data_handler = QueryDocumentPairwiseRankingDataHandler.from_config(
            QueryDocumentPairwiseRankingDataHandler.Config(),
            ModelInputConfig(),
            [],
            featurizer=SimpleFeaturizer.from_config(
                simple_featurizer_config, FeatureConfig()
            ),
        )
        self.file_name = tests_module.test_file(
            "query_document_pairwise_ranking_tiny.tsv"
        )
        self.data_handler.shuffle = False
        self.data_handler.init_metadata_from_path(
            self.file_name, self.file_name, self.file_name
        )

    def create_model(self, shared_rep):
        # shared_rep: do query and response share representation layer?
        metadata = self.data_handler.metadata
        model_config = QueryDocumentPairwiseRankingModel_Deprecated.Config()

        model_config.representation = QueryDocumentPairwiseRankingRep.Config()
        model_config.representation.shared_representations = shared_rep

        model_config.decoder = MLPDecoderQueryResponse.Config()
        model_config.decoder.hidden_dims = [64]
        model_config.output_layer = PairwiseRankingOutputLayer.Config()

        feat_config = ModelInputConfig()
        feat_config.pos_response = WordFeatConfig()
        feat_config.pos_response.embed_dim = 64
        feat_config.neg_response = WordFeatConfig()
        feat_config.query = WordFeatConfig()
        return QueryDocumentPairwiseRankingModel_Deprecated.from_config(
            model_config, feat_config, metadata
        )

    def test_init(self):
        iter = self.data_handler.get_test_iter_from_path(self.file_name, 4)
        self.model_shared_rep.eval()
        for (m_input, _targets, _context) in iter:
            pos_embeddings, neg_embeddings, query_embeddings = map(
                to_numpy, self.model_shared_rep(*m_input)
            )
            self.assertTrue(pos_embeddings.shape[0] == 4)
            self.assertTrue(np.all(np.equal(pos_embeddings[1], neg_embeddings[1])))
            self.assertTrue(np.all(np.equal(pos_embeddings[3], neg_embeddings[3])))
            self.assertFalse(np.all(np.equal(pos_embeddings[0], neg_embeddings[0])))
            self.assertFalse(np.all(np.equal(pos_embeddings[2], neg_embeddings[2])))
            self.assertTrue(np.all(np.equal(query_embeddings[2], query_embeddings[3])))
            self.assertFalse(np.all(np.equal(query_embeddings[1], query_embeddings[2])))

    def test_rep_sharing(self):
        self.assertTrue(
            id(self.model_shared_rep.representation.subrepresentations[0])
            == id(self.model_shared_rep.representation.subrepresentations[1])
        )
        self.assertFalse(
            id(self.model_unshared_rep.representation.subrepresentations[0])
            == id(self.model_unshared_rep.representation.subrepresentations[1])
        )
