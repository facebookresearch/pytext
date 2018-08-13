#!/usr/bin/env python3

from typing import Tuple

from pytext.common.registry import MODEL, component
from pytext.config import ConfigBase
from pytext.config.module_config import LSTMParams, SlotAttentionType
from pytext.models.configs import (
    CharacterEmbeddingConfig,
    DictEmbeddingConfig,
    WordEmbeddingConfig,
)
from pytext.models.crf import CRF
from pytext.models.embeddings.token_embedding import TokenEmbedding
from pytext.models.model import Model
from pytext.models.projections.linear_projection import LinearProjection
from pytext.models.representations.bilstm_slot_attn import BiLSTMSlotAttention


class WordBLSTMConfig(ConfigBase):
    dropout: float = 0.4
    slot_attn_dim: int = 64
    lstm: LSTMParams = LSTMParams()
    slot_attention_type: SlotAttentionType = SlotAttentionType.NO_ATTENTION
    use_crf: bool = False


@component(MODEL, config_cls=WordBLSTMConfig)
class WordBLSTM(Model):
    """
    Word tagging model that uses bidirectional LSTM to represent the document.
    """

    def __init__(
        self,
        model_config: WordBLSTMConfig,
        embedding_config: Tuple[
            WordEmbeddingConfig, DictEmbeddingConfig, CharacterEmbeddingConfig
        ],
        word_class_num: int,
        **kwargs,
    ) -> None:
        super().__init__()

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

        self.projection = LinearProjection(
            self.representation.representation_dim, word_class_num
        )
        self.crf = CRF(word_class_num) if model_config.use_crf else None
