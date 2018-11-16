#!/usr/bin/env python3
from typing import Optional

from .field_config import (
    CharFeatConfig,
    DictFeatConfig,
    DocLabelConfig,
    PretrainedModelEmbeddingConfig,
    WordFeatConfig,
    WordLabelConfig,
)
from .module_config import ModuleConfig
from .pytext_config import ConfigBase


class ModelInputConfig(ModuleConfig):
    word_feat: Optional[WordFeatConfig] = WordFeatConfig()
    dict_feat: Optional[DictFeatConfig] = None
    char_feat: Optional[CharFeatConfig] = None
    pretrained_model_embedding: Optional[PretrainedModelEmbeddingConfig] = None
    seq_word_feat: Optional[WordFeatConfig] = WordFeatConfig()


class TargetConfig(ConfigBase):
    doc_label: DocLabelConfig = DocLabelConfig()
    word_label: WordLabelConfig = WordLabelConfig()


class ModelInput:
    TEXT = "word_feat"
    DICT = "dict_feat"
    CHAR = "char_feat"
    PRETRAINED = "pretrained_model_embedding"
    SEQ = "seq_word_feat"


class Target:
    DOC_LABEL = "doc_label"
    WORD_LABEL = "word_label"


class ExtraField:
    DOC_WEIGHT = "doc_weight"
    WORD_WEIGHT = "word_weight"
    RAW_WORD_LABEL = "raw_word_label"
    TOKEN_RANGE = "token_range"
    INDEX_FIELD = "index_field"
    UTTERANCE = "utterance"
