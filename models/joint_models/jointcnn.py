#!/usr/bin/env python3

from typing import Tuple

from pytext.config import ConfigBase
from pytext.config.module_config import CNNParams
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
from pytext.models.representations.jointcnn_rep import JointCNNRepresentation


class JointCNN(Model):
    """
    Joint intent detection and slot filling model that uses CNNs
    (BiSeqCNNRepresentation and DocNNRepresentation) to represent
    the document.
    Token embedding parameters are shared between the representations.
    """
    class Config(ConfigBase):
        default_doc_loss_weight: float = 0.2
        default_word_loss_weight: float = 0.5
        dropout: float = 0.4
        cnn: CNNParams = CNNParams()
        fwd_bwd_context_len: int = 5
        surrounding_context_len: int = 2
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
        pad_idx: int,
        **kwargs,
    ) -> None:
        super().__init__(model_config)

        self.embedding = TokenEmbedding(*embedding_config)
        self.representation = JointCNNRepresentation(
            model_config.fwd_bwd_context_len,
            model_config.surrounding_context_len,
            self.embedding.embedding_dim,
            1,
            model_config.cnn.kernel_num,
            model_config.cnn.kernel_sizes,
            model_config.dropout,
            pad_idx,
        )
        self.projection = JointModelProjection(
            self.representation.doc_rep.representation_dim,
            self.representation.word_rep.representation_dim,
            doc_class_num,
            word_class_num,
            model_config.use_doc_probs_in_word,
        )
        self.crf = CRF(word_class_num) if model_config.use_crf else None
