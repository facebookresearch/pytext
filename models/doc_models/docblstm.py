#!/usr/bin/env python3

from typing import Tuple, Optional

from pytext.config import ConfigBase
from pytext.config.module_config import LSTMParams, MLPParams
from pytext.models.configs import (
    CharacterEmbeddingConfig,
    DictEmbeddingConfig,
    WordEmbeddingConfig,
)
from pytext.models.embeddings.token_embedding import TokenEmbedding
from pytext.models.model import Model
from pytext.models.projections.mlp_projection import MLPProjection
from pytext.models.representations.bilstm_self_attn import BiLSTMSelfAttention


class DocBLSTM(Model):
    """
    An n-ary document classification model that uses bidirectional LSTM to
    represent the document.
    """

    class Config(ConfigBase):
        dropout: float = 0.4
        # The hidden dimension for the self attention layer
        self_attn_dim: int = 64
        lstm: LSTMParams = LSTMParams()
        mlp: MLPParams = MLPParams()

    def __init__(
        self,
        model_config: Config,
        embedding_config: Tuple[
            WordEmbeddingConfig, DictEmbeddingConfig, CharacterEmbeddingConfig
        ],
        doc_class_num: int,
        **kwargs,
    ) -> None:
        super().__init__(model_config)

        self.embedding = TokenEmbedding(*embedding_config)
        self.representation = BiLSTMSelfAttention(
            self.embedding.embedding_dim,
            model_config.lstm.lstm_dim,
            model_config.lstm.num_layers,
            model_config.dropout,
            model_config.self_attn_dim,
            model_config.lstm.projection_dim,
        )
        self.projection = MLPProjection(
            from_dim=self.representation.representation_dim,
            to_dim=doc_class_num,
            hidden_dims=model_config.mlp.hidden_dims,
        )
