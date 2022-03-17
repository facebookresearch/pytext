#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from enum import Enum
from typing import Tuple

import torch


TORCH_VERSION: Tuple[int, ...] = tuple(
    # Versions could be in the following formats:
    # - 1.9, 1.10 etc.
    # - torch.deploy-1.9, torch.deploy-1.10, etc.
    int(x)
    for x in torch.__version__.split("-")[-1].split(".")[:2]
)


class Token(str):
    def __eq__(self, other):
        # We don't want to compare as equal to actual strings, but we want to behave
        # like a string code-wise. Don't use `is` comparison because we want
        # Token instances created across picklings to equals-compare
        return isinstance(other, Token) and super().__eq__(other)

    def __init__(self, input_str):
        self.str = input_str
        super().__init__()

    __hash__ = str.__hash__


class SpecialTokens:
    UNK = Token("__UNKNOWN__")
    PAD = Token("__PAD__")
    BOS = Token("__BEGIN_OF_SENTENCE__")
    EOS = Token("__END_OF_SENTENCE__")
    BOL = Token("__BEGIN_OF_LIST__")
    EOL = Token("__END_OF_LIST__")
    MASK = Token("__MASK__")
    SELFIE_RAW_IMAGE = Token("__RAW_IMAGE__")
    # BOS and EOS is too long for Byte-level Language Model.
    # Todo: find out conbination of bytes with low-frequency and shorter length
    BYTE_BOS = Token("^")
    BYTE_EOS = Token("#")
    BYTE_SPACE = Token(" ")


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
    TOKEN_INDICES = "token_indices"
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
    CONTEXT_SEQUENCE = "context_sequence"
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
    INDEX = "row_index"
    TASK_NAME = "task_name"


class Stage(Enum):
    TRAIN = "Training"
    EVAL = "Evaluation"
    TEST = "Test"
    OTHERS = "Others"
    PERSONALIZED_EVAL = "Personalized Eval"
    PERSONALIZED_TEST = "Personalized Test"


class RawExampleFieldName:
    ROW_INDEX = "row_index"


# https://www.internalfb.com/code/fbsource/fbcode/nlp_tools/entity_linking/if/entity_type_system.thrift
# Map raw domains to higher level domains from kg_wikidata_nlptools_entityclass table
# Intended to mirror the major wikipedia categories
MATCHA_ENTITIES_MAPPING = {
    "person": "person",
    "fictional_character": "person",
    "politician": "person",
    "child": "person",
    "location": "location",
    "town": "location",
    "city": "location",
    "country": "location",
    "state_province": "location",
    "ocean": "location",
    "lake": "location",
    "island": "location",
    "mountain": "location",
    "deprecated_square": "location",
    "tourist_attraction": "location",
    "retail_location": "location",
    "river": "location",
    "park": "location",
    "road": "location",
    "landmark": "location",
    "winery": "location",
    "vineyard": "location",
    "organization": "organization",
    "band": "organization",
    "sports_team": "organization",
    "sports_league": "organization",
    "tv_channel": "organization",
    "radio_station": "organization",
    "political_party": "organization",
    "record_label": "organization",
    "university": "organization",
    "high_school": "organization",
    "religious_org": "organization",
    "government_org": "organization",
    "publisher": "organization",
    "fictional_school": "organization",
    "restaurant": "organization",
    "supplement_companies": "organization",
    "bar": "organization",
    "brewery": "organization",
    "liquor_store": "organization",
    "tavern": "organization",
    "confederate": "organization",
    "work_of_art": "work_of_art",
    "album": "work_of_art",
    "movie": "work_of_art",
    "song": "work_of_art",
    "tv_show": "work_of_art",
    "literary_work": "work_of_art",
    "video_game": "work_of_art",
    "board_game": "work_of_art",
    "painting": "work_of_art",
    "radio_show": "work_of_art",
    "comic": "work_of_art",
    "musical": "work_of_art",
    "musical_composition": "work_of_art",
    "pole_dance": "work_of_art",
    "erotic_dance": "work_of_art",
    "sexual_content": "work_of_art",
    "gambling": "work_of_art",
    "raffle": "work_of_art",
    "gambling_game": "work_of_art",
    "event": "event",
    "sports_event": "event",
    "music_event": "event",
    "war": "event",
    "holiday": "event",
    "fictional_event": "event",
    "product_service": "product_service",
    "brand": "product_service",
    "drug_brand": "product_service",
    "alcohol_brands": "product_service",
    "software": "product_service",
    "network_service": "product_service",
    "computer": "product_service",
    "vehicle": "product_service",
    "transport_infra": "product_service",
    "animal_products": "product_service",
    "fur": "product_service",
    "financial_product": "product_service",
    "security": "product_service",
    "money_order": "product_service",
    "check": "product_service",
    "virtual_currency": "product_service",
    "credit_card": "product_service",
    "debit_card": "product_service",
    "money": "product_service",
    "currency": "product_service",
    "food_drink": "product_service",
    "drug": "product_service",
    "prescription_drug": "product_service",
    "health_supplement": "product_service",
    "nutrient": "product_service",
    "medical_device": "product_service",
    "contact_lens": "product_service",
    "shake": "product_service",
    "tobacco_products": "product_service",
    "hookah": "product_service",
    "cigarette": "product_service",
    "chewing_tobacco": "product_service",
    "tobacco_pipe": "product_service",
    "tobacco_rolling_machine": "product_service",
    "tobacco": "product_service",
    "bong": "product_service",
    "electronic_cigarette": "product_service",
    "rolling_paper": "product_service",
    "snuff": "product_service",
    "weapon": "product_service",
    "kitchen_knife": "product_service",
    "firearm": "product_service",
    "paintball_gun": "product_service",
    "fireworks": "product_service",
    "explosive": "product_service",
    "pepper_spray": "product_service",
    "fighting_knife": "product_service",
    "electroshock_weapon": "product_service",
    "alcohol": "product_service",
    "illegal_drugs": "product_service",
    "counterfeit_money": "product_service",
    "counterfeit": "product_service",
    "sex_toy": "product_service",
    "kodi": "product_service",
    "spy_camera": "product_service",
    "descrambler": "product_service",
    "radar_detector": "product_service",
    "jammer": "product_service",
    "phone_surveillance": "product_service",
    "giftcard": "product_service",
    "seashell": "product_service",
    "watch": "product_service",
    "jewelry": "product_service",
    "handbag": "product_service",
    "bag": "product_service",
    "shoe": "product_service",
    "clothing": "product_service",
    "wallet": "product_service",
    "belt": "product_service",
    "eyewear": "product_service",
    "perfume": "product_service",
    "cosmetics": "product_service",
    "mobile_phone": "product_service",
    "costume_accessory": "product_service",
    "mobile_phone_accessory": "product_service",
    "tv": "product_service",
    "iwatch": "product_service",
    "playstation": "product_service",
    "cd": "product_service",
    "other": "general_knowledge",
    "language": "general_knowledge",
    "taxon": "general_knowledge",
    "work_position": "general_knowledge",
    "literary_genre": "general_knowledge",
    "music_genre": "general_knowledge",
    "tv_genre": "general_knowledge",
    "religion": "general_knowledge",
    "music_general": "general_knowledge",
    "license": "general_knowledge",
    "certificate": "general_knowledge",
    "political_slogan": "general_knowledge",
    "animal": "general_knowledge",
    "endangered_species": "general_knowledge",
    "healthcare": "general_knowledge",
    "byob": "general_knowledge",
    "illegal": "general_knowledge",
    "false_document": "general_knowledge",
    "human_sexuality": "general_knowledge",
    "intimate_part": "general_knowledge",
    "offensive_material": "general_knowledge",
    "swastica": "general_knowledge",
    "electronic_interference": "general_knowledge",
    "human_body": "general_knowledge",
    "race": "general_knowledge",
    "gender": "general_knowledge",
    "marital_status": "general_knowledge",
    "citizenship": "general_knowledge",
    "nationality": "general_knowledge",
    "sexual_act": "general_knowledge",
    "disease": "general_knowledge",
    "prostitution": "general_knowledge",
    "nazi": "general_knowledge",
    "coupon": "general_knowledge",
    "ticket": "general_knowledge",
    "code": "general_knowledge",
    "job": "general_knowledge",
    "pearl": "general_knowledge",
    "misc": "misc",
}

matcha_entity_raw_domains = sorted(MATCHA_ENTITIES_MAPPING.keys())
matcha_entity_high_level_domains = sorted(set(MATCHA_ENTITIES_MAPPING.values()))
