#!/usr/bin/env python3

from enum import Enum


class DatasetFieldName:
    DOC_LABEL_FIELD = "doc_label"
    WORD_LABEL_FIELD = "word_label"
    UTTERANCE_FIELD = "utterance"
    TEXT_FIELD = "text"
    DICT_FIELD = "dict_feat"
    RAW_DICT_FIELD = "sparsefeat"
    CHAR_FIELD = "char_feat"
    PRETRAINED_MODEL_EMBEDDING = "pretrained_model_embedding"
    DOC_WEIGHT_FIELD = "doc_weight"
    WORD_WEIGHT_FIELD = "word_weight"
    RAW_WORD_LABEL = "raw_word_label"
    TOKEN_RANGE = "token_range"
    INDEX_FIELD = "index_field"
    LANGUAGE_ID_FIELD = "lang"
    SEQ_LENS = "seq_lens"
    TARGET_SEQ_LENS = "target_seq_lens"
    SOURCE_SEQ_FIELD = "source_sequence"
    TARGET_SEQ_FIELD = "target_sequence"


class PackageFileName:
    SERIALIZED_VOCAB = "all_vocab_serialized"
    SERIALIZED_EMBED = "pretrained_embed_pt_serialized"
    RAW_EMBED = "pretrained_embed_raw"


class DFColumn:
    DOC_LABEL = "doc_label"
    WORD_LABEL = "word_label"
    UTTERANCE = "text"
    ALIGNMENT = "alignment"
    DICT_FEAT = "dict_feat"
    RAW_FEATS = "raw_feats"
    MODEL_FEATS = "model_feats"
    DOC_WEIGHT = "doc_weight"
    WORD_WEIGHT = "word_weight"
    TOKEN_RANGE = "token_range"
    LANGUAGE_ID = "lang"
    SOURCE_SEQUENCE = "source_sequence"
    TARGET_SEQUENCE = "target_sequence"
    SOURCE_FEATS = "source_feats"
    TARGET_TOKENS = "target_tokens"


class ConfigeratorPath:
    DEFAULT_TOKENIZER = "configerator:assistant/nlu/default_tokenizer_config"
    LOWER_NO_PUNCT_TOKENIZER = (
        "configerator:assistant/nlu/lower_no_punct_tokenizer_config"
    )
    PREFIX = "configerator:"


class PackageName:
    DEFAULT_VOCAB = "assistant.nlu.vocab"
    DEFAULT_EMBEDDING = "assistant.pretrained_embeddings"


class Padding:
    WORD_LABEL_PAD = "PAD_LABEL"
    WORD_LABEL_PAD_IDX = 0


class VocabMeta:
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"
    EOS_TOKEN = "</s>"
    INIT_TOKEN = "<s>"
    PAD_SEQ = "<pad_seq>"
    EOS_SEQ = "</s_seq>"
    INIT_SEQ = "<s_seq>"


class PredictorInputNames:
    TOKENS_IDS = "tokens_vals"
    TOKENS_LENS = "tokens_lens"
    DICT_FEAT_IDS = "dict_vals"
    DICT_FEAT_WEIGHTS = "dict_weights"
    DICT_FEAT_LENS = "dict_lens"
    CHAR_IDS = "char_vals"
    TOKENS_STR = "tokens_vals_str"
    DICT_FEAT_STR = "dict_vals_str"


class Stage(Enum):
    TRAIN = "Training"
    EVAL = "Evaluation"
    TEST = "Test"
