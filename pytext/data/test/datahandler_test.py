#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest

from pytext.common.constants import DFColumn, VocabMeta
from pytext.config.component import create_featurizer
from pytext.config.doc_classification import ModelInput, ModelInputConfig, TargetConfig
from pytext.config.field_config import FeatureConfig, WordFeatConfig
from pytext.data.data_handler import DataHandler
from pytext.data.doc_classification_data_handler import DocClassificationDataHandler
from pytext.data.featurizer import SimpleFeaturizer
from pytext.utils.test import import_tests_module


tests_module = import_tests_module()


class DataHandlerTest(unittest.TestCase):
    def test_read_from_csv(self):
        file_name = tests_module.test_file("train_data_tiny.tsv")
        columns = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
        ]

        feat = WordFeatConfig(
            vocab_from_all_data=True,
            vocab_from_train_data=False,
            vocab_from_pretrained_embeddings=False,
        )
        featurizer = create_featurizer(
            SimpleFeaturizer.Config(), FeatureConfig(word_feat=feat)
        )
        data_handler = DocClassificationDataHandler.from_config(
            DocClassificationDataHandler.Config(),
            ModelInputConfig(word_feat=feat),
            TargetConfig(),
            featurizer=featurizer,
        )
        data = list(data_handler.read_from_file(file_name, columns))
        for col in columns:
            self.assertTrue(col in data[0], "{} must in the data".format(col))
        self.assertEqual("alarm/modify_alarm", data[0][DFColumn.DOC_LABEL])
        self.assertEqual("16:24:datetime,39:57:datetime", data[0][DFColumn.WORD_LABEL])
        self.assertEqual(
            "change my alarm tomorrow to wake me up 30 minutes earlier",
            data[0][DFColumn.UTTERANCE],
        )
        self.assertEqual("", data[0][DFColumn.DICT_FEAT])

    def test_read_partially_from_csv(self):
        file_name = tests_module.test_file("train_data_tiny.tsv")
        columns = {DFColumn.DOC_LABEL: 0, DFColumn.UTTERANCE: 2}

        feat = WordFeatConfig(
            vocab_from_all_data=True,
            vocab_from_train_data=False,
            vocab_from_pretrained_embeddings=False,
        )
        featurizer = create_featurizer(
            SimpleFeaturizer.Config(), FeatureConfig(word_feat=feat)
        )
        data_handler = DocClassificationDataHandler.from_config(
            DocClassificationDataHandler.Config(),
            ModelInputConfig(word_feat=feat),
            TargetConfig(),
            featurizer=featurizer,
        )
        data = list(data_handler.read_from_file(file_name, columns))
        for col in columns:
            self.assertTrue(col in data[0], "{} must in the data".format(col))
        self.assertEqual("alarm/modify_alarm", data[0][DFColumn.DOC_LABEL])
        self.assertEqual(
            "change my alarm tomorrow to wake me up 30 minutes earlier",
            data[0][DFColumn.UTTERANCE],
        )

    def test_init_feature_metadata(self):
        # Specify data
        feat_name = ModelInput.WORD_FEAT
        train_text = "Hi there you"
        eval_text = ""
        test_text = "Go away"
        pretrained_embedding_file = tests_module.test_file("pretrained_embed_raw")
        pretrained_tokens = {
            "</s>",
            "the",
            "to",
            "and",
            "a",
            "I",
            "you",
            "is",
            "aloha",
            "for",
        }

        # Specify test cases
        test_cases = (
            # Vocab from train / eval / test data
            {
                "feat": WordFeatConfig(
                    vocab_from_all_data=True,
                    vocab_from_train_data=False,
                    vocab_from_pretrained_embeddings=False,
                ),
                "expected_tokens": {
                    "hi",
                    "there",
                    "you",
                    "go",
                    "away",
                    VocabMeta.UNK_TOKEN,
                    VocabMeta.PAD_TOKEN,
                },
                "expected_num_pretrained_tokens": 0,
            },
            # Vocab from train data or pretrained embeddings
            {
                "feat": WordFeatConfig(
                    vocab_from_all_data=False,
                    vocab_from_train_data=True,
                    vocab_from_pretrained_embeddings=True,
                    pretrained_embeddings_path=pretrained_embedding_file,
                    embed_dim=5,
                ),
                "expected_tokens": pretrained_tokens.union(
                    {"hi", "there", VocabMeta.UNK_TOKEN, VocabMeta.PAD_TOKEN}
                ),
                "expected_num_pretrained_tokens": len(pretrained_tokens) + 4,
            },
            # Vocab from limited number of pretrained embeddings
            {
                "feat": WordFeatConfig(
                    vocab_from_all_data=False,
                    vocab_from_train_data=False,
                    vocab_from_pretrained_embeddings=True,
                    pretrained_embeddings_path=pretrained_embedding_file,
                    embed_dim=5,
                    vocab_size=2,
                ),
                "expected_tokens": {
                    "</s>",
                    "the",
                    VocabMeta.UNK_TOKEN,
                    VocabMeta.PAD_TOKEN,
                },
                # special tokens excluded from vocab_size = 2
                "expected_num_pretrained_tokens": 4,
            },
        )

        for case in test_cases:
            # Setup data handler
            featurizer = create_featurizer(
                SimpleFeaturizer.Config(), FeatureConfig(word_feat=case["feat"])
            )
            data_handler = DocClassificationDataHandler.from_config(
                DocClassificationDataHandler.Config(),
                ModelInputConfig(word_feat=case["feat"]),
                TargetConfig(),
                featurizer=featurizer,
            )
            train_data = data_handler.gen_dataset(
                [{"text": train_text}], include_label_fields=False
            )
            eval_data = data_handler.gen_dataset(
                [{"text": eval_text}], include_label_fields=False
            )
            test_data = data_handler.gen_dataset(
                [{"text": test_text}], include_label_fields=False
            )
            data_handler.init_feature_metadata(train_data, eval_data, test_data)

            # Check created vocab
            meta = data_handler.metadata.features[feat_name]
            self.assertEqual(set(meta.vocab.stoi.keys()), case["expected_tokens"])
            if case["expected_num_pretrained_tokens"] == 0:
                self.assertIsNone(meta.pretrained_embeds_weight)
            else:
                self.assertEqual(
                    meta.pretrained_embeds_weight.size(0),
                    case["expected_num_pretrained_tokens"],
                )
