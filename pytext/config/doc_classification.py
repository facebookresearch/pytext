#!/usr/bin/env python3
from .module_config import ModuleConfig
from .pytext_config import ConfigBase
from .field_config import (
    WordFeatConfig,
    DocLabelConfig,
    DictFeatConfig,
    CharFeatConfig,
    PretrainedModelEmbeddingConfig,
)
from typing import Optional


class ModelInputConfig(ModuleConfig):
    word_feat: WordFeatConfig = WordFeatConfig()
    dict_feat: Optional[DictFeatConfig] = None
    char_feat: Optional[CharFeatConfig] = None
    pretrained_model_embedding: Optional[PretrainedModelEmbeddingConfig] = None


class TargetConfig(ConfigBase):
    doc_label: DocLabelConfig = DocLabelConfig()


class ModelInput:
    WORD_FEAT = "word_feat"
    DICT_FEAT = "dict_feat"
    CHAR_FEAT = "char_feat"
    PRETRAINED_MODEL_EMBEDDING = "pretrained_model_embedding"
    SEQ_LENS = "seq_lens"


class Target:
    DOC_LABEL = "doc_label"


class ExtraField:
    RAW_TEXT = "utterance"
    INDEX = "index"
