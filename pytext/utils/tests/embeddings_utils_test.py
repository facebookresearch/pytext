#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import unittest

from pytext.common.constants import DatasetFieldName, VocabMeta
from pytext.config.field_config import EmbedInitStrategy
from pytext.fields import TextFeatureField
from pytext.utils import embeddings_utils


# Contents of these files are exactly same.
RAW_EMBEDDING_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "pretrained_embed_raw"
)
VOCAB_SIZE = 6
EMB_SIZE = 200


def get_text_field():
    text_field = TextFeatureField(DatasetFieldName.TEXT_FIELD)
    text_field.build_vocab([])
    vocab_size = len(text_field.vocab.itos)
    text_field.vocab.itos.extend(["good", "boy"])
    text_field.vocab.stoi.update({"good": vocab_size, "boy": vocab_size + 1})
    print(text_field.vocab.itos)
    return text_field


class PretrainedEmbeddingTest(unittest.TestCase):
    def test_load_pretrained_embeddings(self):
        pretrained_emb = embeddings_utils.PretrainedEmbedding(RAW_EMBEDDING_PATH)

        self.assertEqual(len(pretrained_emb.embed_vocab), VOCAB_SIZE)
        self.assertEqual(pretrained_emb.embed_vocab[0], "</s>")
        self.assertEqual(pretrained_emb.embed_vocab[2], "to")

        self.assertEqual(len(pretrained_emb.stoi), VOCAB_SIZE)
        self.assertEqual(pretrained_emb.stoi["</s>"], 0)
        self.assertEqual(pretrained_emb.stoi["to"], 2)

        self.assertEqual(pretrained_emb.embedding_vectors.size()[0], VOCAB_SIZE)
        self.assertEqual(pretrained_emb.embedding_vectors.size()[1], EMB_SIZE)

    def test_init_pretrained_embeddings(self):
        """Although not a method of PretrainedEmbedding, critical to test."""
        text_field = get_text_field()
        pretrained_emb_tensor = embeddings_utils.init_pretrained_embeddings(
            text_field.vocab.stoi,
            RAW_EMBEDDING_PATH,
            EMB_SIZE,
            VocabMeta.UNK_TOKEN,
            EmbedInitStrategy.ZERO,
            text_field.lower,
        )

        self.assertEqual(pretrained_emb_tensor.size()[0], 4)
        self.assertEqual(pretrained_emb_tensor.size()[1], EMB_SIZE)

        self.assertEqual(text_field.vocab.itos[2], "good")
        self.assertEqual(text_field.vocab.itos[3], "boy")

        self.assertEqual(text_field.vocab.stoi["good"], 2)
        self.assertEqual(text_field.vocab.stoi["boy"], 3)
