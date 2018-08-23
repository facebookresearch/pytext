#!/usr/bin/env python3

from typing import Tuple

from pytext.config import ConfigBase
from pytext.config.module_config import LSTMParams, MLPParams, SlotAttentionType
from pytext.models.configs import (
    CharacterEmbeddingConfig,
    DictEmbeddingConfig,
    WordEmbeddingConfig,
)
from pytext.models.crf import CRF
from pytext.models.embeddings.token_embedding import TokenEmbedding
from pytext.models.model import Model
from pytext.models.projections.mlp_projection import MLPProjection
from pytext.models.representations.bilstm_slot_attn import BiLSTMSlotAttention


class WordBLSTM(Model):
    """
    Word tagging model that uses bidirectional LSTM to represent the document.
    """
    class Config(ConfigBase):
        dropout: float = 0.4
        slot_attn_dim: int = 64
        lstm: LSTMParams = LSTMParams()
        mlp: MLPParams = MLPParams()
        slot_attention_type: SlotAttentionType = SlotAttentionType.NO_ATTENTION
        use_crf: bool = False

    def __init__(
        self,
        model_config: Config,
        embedding_config: Tuple[
            WordEmbeddingConfig, DictEmbeddingConfig, CharacterEmbeddingConfig
        ],
        word_class_num: int,
        **kwargs,
    ) -> None:
        super().__init__(model_config)

        self.embedding = TokenEmbedding(*embedding_config)

        self.representation = BiLSTMSlotAttention(
            self.embedding.embedding_dim,
            model_config.lstm.lstm_dim,
            model_config.lstm.num_layers,
            model_config.dropout,
            model_config.slot_attention_type,
            model_config.slot_attn_dim,
            bidirectional=True,
        )

        self.projection = MLPProjection(
            self.representation.representation_dim,
            model_config.mlp.hidden_dims,
            word_class_num,
        )
        self.crf = CRF(word_class_num) if model_config.use_crf else None
