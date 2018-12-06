#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

import torch
from pytext.config import ConfigBase
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.embeddings import EmbeddingList
from pytext.models.model import Model
from pytext.models.output_layers import ClassificationOutputLayer
from pytext.models.representations.pair_rep import PairRepresentation


class PairClassificationModel(Model):
    """
    A classification model that scores a pair of texts, for example, a model for
    natural language inference.

    The model shares embedding space (so it doesn't support
    pairs of texts where left and right are in different languages). It uses
    bidirectional LSTM or CNN to represent the two documents, and concatenates
    them along with their absolute difference and elementwise product. This
    concatenated pair representation is passed to a multi-layer perceptron to
    decode to label/target space.

    See https://arxiv.org/pdf/1705.02364.pdf for more details.

    It can be instantiated just like any other :class:`~Model`.
    """

    class Config(ConfigBase):
        representation: PairRepresentation.Config = PairRepresentation.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        # TODO: will need to support different output layer for contrastive loss
        output_layer: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )

    @classmethod
    def compose_embedding(cls, sub_embs):
        return EmbeddingList(sub_embs.values(), concat=False)

    def save_modules(self, base_path: str = "", suffix: str = ""):
        super().save_modules(base_path, suffix)

        # Special case to also save the sub-representations separately, if needed.
        for subrep in self.representation.subrepresentations:
            if getattr(subrep.config, "save_path", None):
                path = subrep.config.save_path + suffix
                if base_path:
                    path = os.path.join(base_path, path)
                print(
                    f"Saving state of module {type(subrep).__name__} " f"to {path} ..."
                )
                torch.save(subrep.state_dict(), path)
