#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

from .field_config import (
    CharFeatConfig,
    DictFeatConfig,
    PretrainedModelEmbeddingConfig,
    TargetConfigBase,
    WordFeatConfig,
)
from .module_config import ModuleConfig


class ModelInputConfig(ModuleConfig):
    word_feat: Optional[WordFeatConfig] = WordFeatConfig()
    dict_feat: Optional[DictFeatConfig] = None
    char_feat: Optional[CharFeatConfig] = None
    pretrained_model_embedding: Optional[PretrainedModelEmbeddingConfig] = None
    seq_word_feat: Optional[WordFeatConfig] = WordFeatConfig()


TargetConfig = List[TargetConfigBase]


class ModelInput:
    TEXT = "word_feat"
    DICT = "dict_feat"
    CHAR = "char_feat"
    PRETRAINED = "pretrained_model_embedding"
    SEQ = "seq_word_feat"


class ExtraField:
    DOC_WEIGHT = "doc_weight"
    WORD_WEIGHT = "word_weight"
    RAW_WORD_LABEL = "raw_word_label"
    TOKEN_RANGE = "token_range"
    UTTERANCE = "utterance"
