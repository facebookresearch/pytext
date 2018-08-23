#!/usr/bin/env python3

from typing import Tuple

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
from pytext.models.projections.joint_model_projection import (
    JointModelProjection
)
from pytext.models.representations.jointblstm_rep import (
    JointBLSTMRepresentation
)


class JointBLSTM(Model):
    class Config(ConfigBase):
        default_doc_loss_weight: float = 0.2
        default_word_loss_weight: float = 0.5
        dropout: float = 0.4
        self_attn_dim: int = 64
        lstm: LSTMParams = LSTMParams()
        slot_attention_type: SlotAttentionType = SlotAttentionType.NO_ATTENTION
        use_crf: bool = False
        use_doc_probs_in_word: bool = False

    def __init__(
        self,
        model_config: Config,
        embedding_config: Tuple[
            WordEmbeddingConfig, DictEmbeddingConfig, CharacterEmbeddingConfig
        ],
        doc_class_num: int,
        word_class_num: int,
        **kwargs,
    ) -> None:
        super().__init__(model_config)

        self.embedding = TokenEmbedding(*embedding_config)
        self.representation = JointBLSTMRepresentation(
            self.embedding.embedding_dim,
            model_config.lstm.lstm_dim,
            model_config.lstm.num_layers,
            True,  # TODO: Add config support for this
            model_config.dropout,
            model_config.self_attn_dim,
            model_config.self_attn_dim,
            model_config.slot_attention_type,
            doc_class_num,
        )
        self.projection = JointModelProjection(
            2 * model_config.lstm.lstm_dim,
            2 * model_config.lstm.lstm_dim,
            doc_class_num,
            word_class_num,
            model_config.use_doc_probs_in_word,
        )
        self.crf = CRF(word_class_num) if model_config.use_crf else None
