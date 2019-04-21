#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

from .field_config import (
    CharFeatConfig,
    ContextualTokenEmbeddingConfig,
    DictFeatConfig,
    FloatVectorConfig,
    TargetConfigBase,
    WordFeatConfig,
)
from .module_config import ModuleConfig


class ModelInputConfig(ModuleConfig):
    word_feat: Optional[WordFeatConfig] = WordFeatConfig()
    dict_feat: Optional[DictFeatConfig] = None
    char_feat: Optional[CharFeatConfig] = None
    contextual_token_embedding: Optional[ContextualTokenEmbeddingConfig] = None
    seq_word_feat: Optional[WordFeatConfig] = WordFeatConfig()
    dense_feat: Optional[FloatVectorConfig] = None


TargetConfig = List[TargetConfigBase]


class ModelInput:
    TEXT = "word_feat"
    DICT = "dict_feat"
    CHAR = "char_feat"
    CONTEXTUAL_TOKEN_EMBEDDING = "contextual_token_embedding"
    SEQ = "seq_word_feat"
    DENSE = "dense_feat"


class ExtraField:
    DOC_WEIGHT = "doc_weight"
    WORD_WEIGHT = "word_weight"
    RAW_WORD_LABEL = "raw_word_label"
    TOKEN_RANGE = "token_range"
    UTTERANCE = "utterance"
