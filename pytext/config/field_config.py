#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from enum import Enum
from typing import Dict, List, Optional, Union

from pytext.common.constants import DatasetFieldName

from .module_config import CNNParams, ModuleConfig, PoolingType
from .pytext_config import ConfigBase


class EmbedInitStrategy(Enum):
    RANDOM = "random"
    ZERO = "zero"


class Target:
    DOC_LABEL = "doc_label"
    TARGET_LOGITS_FIELD = "target_logit"
    TARGET_PROB_FIELD = "target_prob"
    TARGET_LABEL_FIELD = "target_label"


class WordFeatConfig(ModuleConfig):
    embed_dim: int = 100
    freeze: bool = False  # only freezes embedding lookup, not MLP layers
    embedding_init_strategy: EmbedInitStrategy = EmbedInitStrategy.RANDOM
    embedding_init_range: Optional[List[float]] = None
    export_input_names: List[str] = ["tokens_vals"]
    pretrained_embeddings_path: str = ""
    vocab_file: str = ""
    #: If `pretrained_embeddings_path` and `vocab_from_pretrained_embeddings` are set,
    #: only the first `vocab_size` tokens in the file will be added to the vocab.
    vocab_size: int = 0
    vocab_from_train_data: bool = True  # add tokens from train data to vocab
    vocab_from_all_data: bool = False  # add tokens from train, eval, test data to vocab
    # add tokens from pretrained embeddings to vocab
    vocab_from_pretrained_embeddings: bool = False
    lowercase_tokens: bool = True
    min_freq: int = 1
    mlp_layer_dims: Optional[List[int]] = []


class DictFeatConfig(ModuleConfig):
    embed_dim: int = 100
    sparse: bool = False
    pooling: PoolingType = PoolingType.MEAN
    export_input_names: List[str] = ["dict_vals", "dict_weights", "dict_lens"]
    vocab_from_train_data: bool = True


class CharFeatConfig(ModuleConfig):
    embed_dim: int = 100
    sparse: bool = False
    cnn: CNNParams = CNNParams()
    highway_layers: int = 0
    projection_dim: Optional[int] = None
    export_input_names: List[str] = ["char_vals"]
    vocab_from_train_data: bool = True
    max_word_length: int = 20
    min_freq: int = 1


class ContextualTokenEmbeddingConfig(ConfigBase):
    embed_dim: int = 0
    model_paths: Optional[Dict[str, str]] = None
    export_input_names: List[str] = ["contextual_token_embedding"]


class FloatVectorConfig(ConfigBase):
    dim: int = 0  # Dimension of the vector in the dataset.
    export_input_names: List[str] = ["float_vec_vals"]
    dim_error_check: bool = False  # should we error check dims b/w config and data


class FeatureConfig(ModuleConfig):  # type: ignore
    word_feat: WordFeatConfig = WordFeatConfig()
    seq_word_feat: Optional[WordFeatConfig] = None
    dict_feat: Optional[DictFeatConfig] = None
    char_feat: Optional[CharFeatConfig] = None
    dense_feat: Optional[FloatVectorConfig] = None
    contextual_token_embedding: Optional[ContextualTokenEmbeddingConfig] = None


class WordLabelConfig(ConfigBase):
    # Transform sequence labels to BIO format
    use_bio_labels: bool = False
    export_output_names: List[str] = ["word_scores"]
    _name = DatasetFieldName.WORD_LABEL_FIELD


class DocLabelConfig(ConfigBase):
    export_output_names: List[str] = ["doc_scores"]
    label_weights: Dict[str, float] = {}
    _name = DatasetFieldName.DOC_LABEL_FIELD
    target_prob: bool = False


TargetConfigBase = Union[DocLabelConfig, WordLabelConfig]
