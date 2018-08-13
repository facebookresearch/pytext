#!/usr/bin/env python3

from typing import Tuple

from pytext.common.registry import MODEL, component
from pytext.config.pytext_config import ConfigBase
from pytext.config.module_config import CNNParams
from pytext.models.configs import (
    CharacterEmbeddingConfig,
    DictEmbeddingConfig,
    WordEmbeddingConfig,
)
from pytext.models.embeddings.token_embedding import TokenEmbedding
from pytext.models.model import Model
from pytext.models.projections.linear_projection import LinearProjection
from pytext.models.representations.docnn import DocNNRepresentation


class DocNNConfig(ConfigBase):
    dropout: float = 0.4
    cnn: CNNParams = CNNParams()


@component(MODEL, config_cls=DocNNConfig)
class DocNN(Model):
    """
    An n-ary document classification model that uses CNN to
    represent the document.
    """

    def __init__(
        self,
        model_config: DocNNConfig,
        embedding_config: Tuple[
            WordEmbeddingConfig, DictEmbeddingConfig, CharacterEmbeddingConfig
        ],
        doc_class_num: int,
        pad_idx: int,
        **kwargs,
    ) -> None:
        super().__init__()

        self.embedding = TokenEmbedding(*embedding_config)
        self.representation = DocNNRepresentation(
            self.embedding.embedding_dim,
            model_config.cnn.kernel_num,
            model_config.cnn.kernel_sizes,
            model_config.dropout,
            pad_idx,
        )
        self.projection = LinearProjection(
            self.representation.representation_dim, doc_class_num
        )
