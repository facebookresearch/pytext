#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from typing import List, Tuple

import torch
from pytext.config import ConfigBase
from pytext.models.representations.transformer_sentence_encoder_base import (
    TransformerSentenceEncoderBase,
)
from pytext.utils.file_io import PathManager
from pytorch_pretrained_bert.modeling import BertConfig, BertModel


class HuggingFaceBertSentenceEncoder(TransformerSentenceEncoderBase):
    """
    Generate sentence representation using the open source HuggingFace BERT
    model. This class implements loading the model weights from a
    pre-trained model file.
    """

    class Config(TransformerSentenceEncoderBase.Config, ConfigBase):
        bert_cpt_dir: str = "/mnt/vol/nlp_technologies/bert/uncased_L-12_H-768_A-12/"
        load_weights: bool = True

    def __init__(
        self, config: Config, output_encoded_layers: bool, *args, **kwargs
    ) -> None:
        super().__init__(config, output_encoded_layers=output_encoded_layers)
        # Load config
        config_file = os.path.join(config.bert_cpt_dir, "bert_config.json")
        bert_config = BertConfig.from_json_file(config_file)
        print("Bert model config {}".format(bert_config))
        # Instantiate model.
        model = BertModel(bert_config)
        weights_path = os.path.join(config.bert_cpt_dir, "pytorch_model.bin")
        # load pre-trained weights if weights_path exists
        if config.load_weights and PathManager.isfile(weights_path):
            state_dict = torch.load(weights_path)

            missing_keys: List[str] = []
            unexpected_keys: List[str] = []
            error_msgs: List[str] = []
            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            def load(module, prefix=""):
                local_metadata = (
                    {} if metadata is None else metadata.get(prefix[:-1], {})
                )
                module._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    True,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + ".")

            load(model, prefix="" if hasattr(model, "bert") else "bert.")
            if len(missing_keys) > 0:
                print(
                    "Weights of {} not initialized from pretrained model: {}".format(
                        model.__class__.__name__, missing_keys
                    )
                )
            if len(unexpected_keys) > 0:
                print(
                    "Weights from pretrained model not used in {}: {}".format(
                        model.__class__.__name__, unexpected_keys
                    )
                )

        self.bert = model

    def _encoder(self, input_tuple: Tuple[torch.Tensor, ...]):
        tokens, pad_mask, segment_labels, _ = input_tuple
        return self.bert(tokens, segment_labels, pad_mask)

    def _embedding(self):
        # used to tie weights in MaskedLM model
        return self.bert.embeddings.word_embeddings
