#!/usr/bin/env python3

from typing import Tuple

from pytext.common.registry import MODEL, component
from pytext.config import ConfigBase
from pytext.config.module_config import LSTMParams
from pytext.models.configs import (
    CharacterEmbeddingConfig,
    DictEmbeddingConfig,
    WordEmbeddingConfig,
)
from pytext.models.embeddings.token_embedding import TokenEmbedding
from pytext.models.model import Model
from pytext.models.projections.linear_projection import LinearProjection
from pytext.models.representations.bilstm_self_attn import BiLSTMSelfAttention


class DocBLSTMConfig(ConfigBase):
    dropout: float = 0.4
    # The hidden dimension for the self attention layer
    self_attn_dim: int = 64
    lstm: LSTMParams = LSTMParams()


@component(MODEL, config_cls=DocBLSTMConfig)
class DocBLSTM(Model):
    """
    An n-ary document classification model that uses bidirectional LSTM to
    represent the document.
    """

    def __init__(
        self,
        model_config: DocBLSTMConfig,
        embedding_config: Tuple[
            WordEmbeddingConfig, DictEmbeddingConfig, CharacterEmbeddingConfig
        ],
        doc_class_num: int,
        **kwargs,
    ) -> None:
        super().__init__()

        self.embedding = TokenEmbedding(*embedding_config)
        self.representation = BiLSTMSelfAttention(
            self.embedding.embedding_dim,
            model_config.lstm.lstm_dim,
            model_config.lstm.num_layers,
            model_config.dropout,
            model_config.self_attn_dim,
        )
        self.projection = LinearProjection(
            self.representation.representation_dim, doc_class_num
        )
