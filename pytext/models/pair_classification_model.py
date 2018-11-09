#!/usr/bin/env python3

import os
from pytext.utils import cuda_utils
import torch
from pytext.config import ConfigBase
from pytext.models.embeddings import EmbeddingList
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.model import Model, DataParallelModel
from pytext.models.output_layer import ClassificationOutputLayer
from pytext.models.representations.tuple_rep import TupleRepresentation
from pytext.models.module import create_module


class PairClassificationModel(Model):
    """Pair classification model

    The model takes two sets of tokens (left and right), calculates their
    representations, and passes them to the decoder along with their
    absolute difference and elementwise product, all concatenated. Used for e.g.
    natural language inference.

    See e.g. <https://arxiv.org/pdf/1705.02364.pdf>.
    """

    class Config(ConfigBase):
        representation: TupleRepresentation.Config = TupleRepresentation.Config()
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
        for i, subrep in enumerate(self.representation.subrepresentations):
            if getattr(subrep.config, "save_path", None):
                path = subrep.config.save_path + "-" + str(i) + suffix
                if base_path:
                    path = os.path.join(base_path, path)
                print(
                    f"Saving state of module {type(subrep).__name__} " f"to {path} ..."
                )
                torch.save(subrep.state_dict(), path)
