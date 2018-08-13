#!/usr/bin/env python3

from typing import Tuple

from pytext.common.registry import MODEL, component
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
from pytext.models.projections.linear_projection import LinearProjection
from pytext.models.representations.biseqcnn import BSeqCNNRepresentation


class WordCNNConfig(ConfigBase):
    dropout: float = 0.4
    cnn = CNNParams()
    fwd_bwd_context_len: int = 5
    surrounding_context_len: int = 2
    use_crf: bool = False


@component(MODEL, config_cls=WordCNNConfig)
class WordCNN(Model):
    """
    Word tagging model that uses CNN to represent the document.
    Specifically it uses BidirectionalSeqCNN for sentence representation.
    """

    def __init__(
        self,
        model_config: WordCNNConfig,
        embedding_config: Tuple[
            WordEmbeddingConfig, DictEmbeddingConfig, CharacterEmbeddingConfig
        ],
        word_class_num: int,
        pad_idx: int,
        **kwargs,
    ) -> None:
        super().__init__()

        in_channels = 1
        out_channels = model_config.cnn.kernel_num
        kernel_sizes = model_config.cnn.kernel_sizes

        fwd_bwd_ctxt_len = model_config.fwd_bwd_context_len
        surr_ctxt_len = model_config.surrounding_context_len
        ctxt_pad_len = max(fwd_bwd_ctxt_len, surr_ctxt_len)

        self.embedding = TokenEmbedding(*embedding_config)
        self.representation = BSeqCNNRepresentation(
            fwd_bwd_ctxt_len,
            surr_ctxt_len,
            ctxt_pad_len,
            pad_idx,
            self.embedding.embedding_dim,
            in_channels,
            out_channels,
            kernel_sizes,
        )
        self.projection = LinearProjection(
            self.representation.representation_dim, word_class_num
        )
        self.crf = CRF(word_class_num) if model_config.use_crf else None
