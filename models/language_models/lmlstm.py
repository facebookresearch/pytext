#!/usr/bin/env python3
from pytext.config import ConfigBase
from pytext.data import CommonMetadata
from pytext.models.model import Model
from pytext.models.projections.mlp_projection import MLPProjection
from pytext.models.representations.bilstm_self_attn import BiLSTMSelfAttention


class LMLSTM(Model):
    """
    A word-level language model that uses LSTM to represent the document
    """
    class Config(ConfigBase):
        repr_config: BiLSTMSelfAttention.Config = BiLSTMSelfAttention.Config(
            self_attn_dim=0, bidirectional=False
        )
        proj_config: MLPProjection.Config = MLPProjection.Config()
        tied_weights: bool = False

    @classmethod
    def from_config(cls, model_config, feat_config, metadata: CommonMetadata):
        model = super().from_config(model_config, feat_config, metadata)
        print(model_config)
        if model_config.tied_weights:
            if not feat_config.word_feat:
                raise ValueError(
                    "Word embeddings must be used when enabling tied weights"
                )
            elif (
                model.embedding.word_embed.embedding_dim
                != model.representation.representation_dim
            ):
                print(model.embedding.word_embed.embedding_dim)
                print(model.representation.representation_dim)
                raise ValueError(
                    "Embedding dimension must be same as representation "
                    "dimesnions when using tied weights"
                )
            model.projection.get_projection()[
                0
            ].weight = model.embedding.word_embed.weight
        return model
