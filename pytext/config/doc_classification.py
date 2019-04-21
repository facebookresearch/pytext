#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional

from .field_config import (
    CharFeatConfig,
    ContextualTokenEmbeddingConfig,
    DictFeatConfig,
    DocLabelConfig,
    FloatVectorConfig,
    WordFeatConfig,
)
from .module_config import ModuleConfig


class ModelInputConfig(ModuleConfig):
    word_feat: WordFeatConfig = WordFeatConfig()
    dict_feat: Optional[DictFeatConfig] = None
    char_feat: Optional[CharFeatConfig] = None
    contextual_token_embedding: Optional[ContextualTokenEmbeddingConfig] = None
    dense_feat: Optional[FloatVectorConfig] = None


TargetConfig = DocLabelConfig


class ModelInput:
    WORD_FEAT = "word_feat"
    DICT_FEAT = "dict_feat"
    CHAR_FEAT = "char_feat"
    CONTEXTUAL_TOKEN_EMBEDDING = "contextual_token_embedding"
    SEQ_LENS = "seq_lens"
    DENSE_FEAT = "dense_feat"


class ExtraField:
    RAW_TEXT = "utterance"
