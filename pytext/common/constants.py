#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum


class DatasetFieldName:
    DOC_LABEL_FIELD = "doc_label"
    WORD_LABEL_FIELD = "word_label"
    UTTERANCE_FIELD = "utterance"
    TEXT_FIELD = "word_feat"
    SEQ_FIELD = "seq_word_feat"
    DICT_FIELD = "dict_feat"
    RAW_DICT_FIELD = "sparsefeat"
    CHAR_FIELD = "char_feat"
    DENSE_FIELD = "dense_feat"
    CONTEXTUAL_TOKEN_EMBEDDING = "contextual_token_embedding"
    DOC_WEIGHT_FIELD = "doc_weight"
    WORD_WEIGHT_FIELD = "word_weight"
    RAW_WORD_LABEL = "raw_word_label"
    TOKEN_RANGE = "token_range"
    TOKENS = "tokens"
    LANGUAGE_ID_FIELD = "lang"
    SEQ_LENS = "seq_lens"
    TARGET_SEQ_LENS = "target_seq_lens"
    RAW_SEQUENCE = "raw_sequence"
    SOURCE_SEQ_FIELD = "source_sequence"
    TARGET_SEQ_FIELD = "target_sequence"
    NUM_TOKENS = "num_tokens"


class PackageFileName:
    SERIALIZED_EMBED = "pretrained_embed_pt_serialized"
    RAW_EMBED = "pretrained_embed_raw"


class DFColumn:
    DOC_LABEL = "doc_label"
    WORD_LABEL = "word_label"
    UTTERANCE = "text"
    ALIGNMENT = "alignment"
    DICT_FEAT = "dict_feat"
    DENSE_FEAT = "dense_feat"
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
    SEQLOGICAL = "seqlogical"
    TARGET_PROBS = "target_probs"
    TARGET_LOGITS = "target_logits"
    TARGET_LABELS = "target_labels"


class Padding:
    WORD_LABEL_PAD = "PAD_LABEL"
    WORD_LABEL_PAD_IDX = 0
    DEFAULT_LABEL_PAD_IDX = -1


class VocabMeta:
    UNK_TOKEN = "<unk>"
    UNK_NUM_TOKEN = f"{UNK_TOKEN}-NUM"
    PAD_TOKEN = "<pad>"
    EOS_TOKEN = "</s>"
    INIT_TOKEN = "<s>"
    PAD_SEQ = "<pad_seq>"
    EOS_SEQ = "</s_seq>"
    INIT_SEQ = "<s_seq>"


class BatchContext:
    IGNORE_LOSS = "ignore_loss"
    INDEX = "index"
    TASK_NAME = "task_name"


class Stage(Enum):
    TRAIN = "Training"
    EVAL = "Evaluation"
    TEST = "Test"


class RawExampleFieldName:
    ROW_INDEX = "row_index"
