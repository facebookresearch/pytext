#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Optional

from .field_config import (
    CharFeatConfig,
    DictFeatConfig,
    DocLabelConfig,
    PretrainedModelEmbeddingConfig,
    WordFeatConfig,
)
from .module_config import ModuleConfig


class ModelInputConfig(ModuleConfig):
    word_feat: WordFeatConfig = WordFeatConfig()
    dict_feat: Optional[DictFeatConfig] = None
    char_feat: Optional[CharFeatConfig] = None
    pretrained_model_embedding: Optional[PretrainedModelEmbeddingConfig] = None


TargetConfig = DocLabelConfig


class ModelInput:
    WORD_FEAT = "word_feat"
    DICT_FEAT = "dict_feat"
    CHAR_FEAT = "char_feat"
    PRETRAINED_MODEL_EMBEDDING = "pretrained_model_embedding"
    SEQ_LENS = "seq_lens"


class Target:
    DOC_LABEL = "doc_label"
    TARGET_LOGITS_FIELD = "target_logit"
    TARGET_PROB_FIELD = "target_prob"
    TARGET_LABEL_FIELD = "target_label"


class ExtraField:
    RAW_TEXT = "utterance"
