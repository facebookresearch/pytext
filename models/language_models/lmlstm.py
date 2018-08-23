#!/usr/bin/env python3

from typing import Tuple

from pytext.common.constants import PredictorInputNames
from pytext.config import ConfigBase
from pytext.config.module_config import LSTMParams, MLPParams
from pytext.data.data_handler import COMMON_META
from pytext.models.configs import (
    CharacterEmbeddingConfig,
    DictEmbeddingConfig,
    WordEmbeddingConfig,
)
from pytext.models.embeddings.token_embedding import TokenEmbedding
from pytext.models.model import Model
from pytext.models.projections.mlp_projection import MLPProjection
from pytext.models.representations.bilstm_self_attn import BiLSTMSelfAttention


class LMLSTM(Model):
    """
    A word-level language model that uses LSTM to represent the document
    """
    class Config(ConfigBase):
        dropout: float = 0.4
        lstm: LSTMParams = LSTMParams()
        mlp: MLPParams = MLPParams()
        tied_weights: bool = False

    def __init__(
        self,
        model_config: Config,
        embedding_config: Tuple[
            WordEmbeddingConfig, DictEmbeddingConfig, CharacterEmbeddingConfig
        ],
        **kwargs,
    ) -> None:
        super().__init__(model_config)
        num_classes = len(
            kwargs[COMMON_META.FEATURE_VOCABS].get(PredictorInputNames.TOKENS_IDS).itos
        )

        self.embedding = TokenEmbedding(*embedding_config)
        self.representation = BiLSTMSelfAttention(
            self.embedding.embedding_dim,
            model_config.lstm.lstm_dim,
            model_config.lstm.num_layers,
            model_config.dropout,
            self_attn_dim=0,
            bidirectional=False,
        )
        self.projection = MLPProjection(
            self.representation.representation_dim,
            model_config.mlp.hidden_dims,
            num_classes,
        )
        if model_config.tied_weights is True:
            if not embedding_config[0]:
                raise ValueError(
                    "Word embeddings must be used when enabling tied weights"
                )
            elif (
                embedding_config[0].embedding_dim
                != self.representation.representation_dim
            ):
                raise ValueError(
                    "Embedding dimension must be same as representation "
                    "dimesnions when using tied weights"
                )
            self.projection.get_projection()[
                0
            ].weight = self.embedding.word_embed.weight
