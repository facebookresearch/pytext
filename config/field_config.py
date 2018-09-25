#!/usr/bin/env python3
from enum import Enum
from typing import Dict, List, Optional

from .module_config import CNNParams, PoolingType
from .pytext_config import ConfigBase


class EmbedInitStrategy(Enum):
    RANDOM = "random"
    ZERO = "zero"


class WordFeatConfig(ConfigBase):
    embed_dim: int = 100
    sparse: bool = False
    freeze: bool = False
    embedding_init_strategy: EmbedInitStrategy = EmbedInitStrategy.RANDOM
    embedding_init_range: Optional[List[float]] = None
    export_input_names: List[str] = ["tokens_vals", "tokens_lens"]
    pretrained_embeddings_path: str = ""
    vocab_file: str = ""
    vocab_size: int = 0
    vocab_from_train_data: bool = True


class DictFeatConfig(ConfigBase):
    embed_dim: int = 100
    sparse: bool = False
    pooling: PoolingType = PoolingType.MEAN
    export_input_names: List[str] = ["dict_vals", "dict_weights", "dict_lens"]
    vocab_from_train_data: bool = True


class CharFeatConfig(ConfigBase):
    embed_dim: int = 100
    sparse: bool = False
    cnn: CNNParams = CNNParams()
    vocab_from_train_data: bool = True


class PretrainedModelEmbeddingConfig(ConfigBase):
    embed_dim: int
    model_file_paths: Dict[str, str]


class CapFeatConfig(ConfigBase):
    embed_dim: int = 100


class FeatureConfig(ConfigBase):
    word_feat: WordFeatConfig = WordFeatConfig()
    dict_feat: Optional[DictFeatConfig] = None
    char_feat: Optional[CharFeatConfig] = None
    cap_feat: Optional[CapFeatConfig] = None
    pretrained_model_embedding: Optional[PretrainedModelEmbeddingConfig] = None


class WordLabelConfig(ConfigBase):
    # Transform sequence labels to BIO format
    use_bio_labels: bool = False
    export_output_names: List[str] = ["word_scores"]


class DocLabelConfig(ConfigBase):
    export_output_names: List[str] = ["doc_scores"]


class LabelConfig(ConfigBase):
    doc_label: Optional[DocLabelConfig] = None
    word_label: Optional[WordLabelConfig] = None
