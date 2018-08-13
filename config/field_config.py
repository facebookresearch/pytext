#!/usr/bin/env python3
from enum import Enum
from typing import List, Optional

from .module_config import CNNParams, PoolingType
from .pytext_config import ConfigBase


class EmbedInitStrategy(Enum):
    RANDOM = "random"
    ZERO = "zero"


class WordFeatConfig(ConfigBase):
    embed_dim: int = 100
    sparse: bool = False
    freeze: bool = False
    embed_init_strategy: EmbedInitStrategy = EmbedInitStrategy.RANDOM
    export_input_names: List[str] = ["tokens_vals", "tokens_lens"]


class DictFeatConfig(ConfigBase):
    embed_dim: int = 100
    pooling: PoolingType = PoolingType.MEAN
    export_input_names: List[str] = ["dict_vals", "dict_weights", "dict_lens"]


class CharFeatConfig(ConfigBase):
    embed_dim: int = 100
    cnn: CNNParams = CNNParams()


class CapFeatConfig(ConfigBase):
    embed_dim: int = 100


class FeatureConfig(ConfigBase):
    word_feat: WordFeatConfig = WordFeatConfig()
    dict_feat: Optional[DictFeatConfig] = None
    char_feat: Optional[CharFeatConfig] = None
    cap_feat: Optional[CapFeatConfig] = None


class WordLabelConfig(ConfigBase):
    # Transform sequence labels to BIO format
    use_bio_labels: bool = False
    export_output_names: List[str] = ["word_scores"]


class DocLabelConfig(ConfigBase):
    export_output_names: List[str] = ["doc_scores"]


class LabelConfig(ConfigBase):
    doc_label: Optional[DocLabelConfig] = None
    word_label: Optional[WordLabelConfig] = None
