#!/usr/bin/env python3
from .field_config import DocLabelConfig, WordFeatConfig
from .module_config import ModuleConfig
from .pytext_config import ConfigBase


class ModelInputConfig(ModuleConfig):
    text1: WordFeatConfig = WordFeatConfig()
    text2: WordFeatConfig = WordFeatConfig()


class TargetConfig(ConfigBase):
    doc_label: DocLabelConfig = DocLabelConfig()


class ModelInput:
    TEXT1 = "text1"
    TEXT2 = "text2"


class Target:
    DOC_LABEL = "doc_label"


class ExtraField:
    UTTERANCE_PAIR = "utterance"
