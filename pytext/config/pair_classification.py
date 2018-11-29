#!/usr/bin/env python3
from .field_config import DocLabelConfig, WordFeatConfig
from .module_config import ModuleConfig


class ModelInputConfig(ModuleConfig):
    text1: WordFeatConfig = WordFeatConfig()
    text2: WordFeatConfig = WordFeatConfig()


TargetConfig = DocLabelConfig


class ModelInput:
    TEXT1 = "text1"
    TEXT2 = "text2"


class ExtraField:
    UTTERANCE_PAIR = "utterance"
