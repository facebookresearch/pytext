#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional

from .field_config import (
    CharFeatConfig,
    DictFeatConfig,
    DocLabelConfig,
    FloatVectorConfig,
    PretrainedModelEmbeddingConfig,
    WordFeatConfig,
)
from .module_config import ModuleConfig


class ModelInputConfig(ModuleConfig):
    word_feat: WordFeatConfig = WordFeatConfig()
    dict_feat: Optional[DictFeatConfig] = None
    char_feat: Optional[CharFeatConfig] = None
    pretrained_model_embedding: Optional[PretrainedModelEmbeddingConfig] = None
    dense_feat: Optional[FloatVectorConfig] = None


TargetConfig = DocLabelConfig


class ModelInput:
    WORD_FEAT = "word_feat"
    DICT_FEAT = "dict_feat"
    CHAR_FEAT = "char_feat"
    PRETRAINED_MODEL_EMBEDDING = "pretrained_model_embedding"
    SEQ_LENS = "seq_lens"
    DENSE_FEAT = "dense_feat"


class ExtraField:
    RAW_TEXT = "utterance"
