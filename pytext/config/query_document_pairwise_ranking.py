#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .field_config import WordFeatConfig
from .module_config import ModuleConfig


class ModelInputConfig(ModuleConfig):
    pos_response: WordFeatConfig = WordFeatConfig()
    neg_response: WordFeatConfig = WordFeatConfig()
    query: WordFeatConfig = WordFeatConfig()


class ModelInput:
    QUERY = "query"
    POS_RESPONSE = "pos_response"
    NEG_RESPONSE = "neg_response"
