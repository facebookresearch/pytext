#!/usr/bin/env python3

from typing import Tuple
from pytext.config.field_config import FeatureConfig
from .embedding_config import (
    CharacterEmbeddingConfig,
    DictEmbeddingConfig,
    WordEmbeddingConfig,
)


def gen_embedding_config(
    feat_config: FeatureConfig,
    embed_num,
    unk_idx,
    pretrained_embeds_weight=None,
    dict_embed_num=0,
    char_embed_num=0,
    **kwargs
) -> Tuple[WordEmbeddingConfig, DictEmbeddingConfig, CharacterEmbeddingConfig]:
    """Generate appropriate embedding config objects by parsing PyTextConfig."""
    word_emb_config = (
        WordEmbeddingConfig(
            embed_num,
            feat_config.word_feat.embed_dim,
            pretrained_embeds_weight,
            unk_idx,
            feat_config.word_feat.sparse,
            feat_config.word_feat.freeze,
        )
        if embed_num
        else None
    )

    dict_emb_config = (
        DictEmbeddingConfig(
            dict_embed_num,
            feat_config.dict_feat.embed_dim,
            feat_config.dict_feat.pooling,
        )
        if feat_config.dict_feat
        else None
    )
    char_emb_config = (
        CharacterEmbeddingConfig(
            char_embed_num,
            feat_config.char_feat.embed_dim,
            feat_config.char_feat.cnn.kernel_num,
            feat_config.char_feat.cnn.kernel_sizes,
        )
        if feat_config.char_feat
        else None
    )

    return word_emb_config, dict_emb_config, char_emb_config
