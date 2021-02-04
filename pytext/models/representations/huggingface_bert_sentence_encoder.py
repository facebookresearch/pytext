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
from pytext.utils.usage import log_class_usage
from transformers.modeling_bert import BertConfig, BertModel


class HuggingFaceBertSentenceEncoder(TransformerSentenceEncoderBase):
    """
    Generate sentence representation using the open source HuggingFace BERT
    model. This class implements loading the model weights from a
    pre-trained model file.
    """

    class Config(TransformerSentenceEncoderBase.Config, ConfigBase):
        bert_cpt_dir: str = (
            "manifold://nlp_technologies/tree/huggingface-models/bert-base-uncased/"
        )
        load_weights: bool = True

    def __init__(
        self, config: Config, output_encoded_layers: bool, *args, **kwargs
    ) -> None:
        super().__init__(config, output_encoded_layers=output_encoded_layers)
        # Load config
        config_file = os.path.join(config.bert_cpt_dir, "config.json")
        local_config_path = PathManager.get_local_path(config_file)
        bert_config = BertConfig.from_json_file(local_config_path)
        print("Bert model config {}".format(bert_config))
        # Instantiate model.
        model = BertModel(bert_config)
        weights_path = os.path.join(config.bert_cpt_dir, "pytorch_model.bin")
        # load pre-trained weights if weights_path exists
        if config.load_weights and PathManager.isfile(weights_path):
            with PathManager.open(weights_path, "rb") as fd:
                state_dict = torch.load(fd)

            missing_keys: List[str] = []
            unexpected_keys: List[str] = []
            error_msgs: List[str] = []
            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            for key in list(state_dict.keys()):
                new_key = None
                if key.endswith("LayerNorm.gamma"):  # compatibility with v0.5 models
                    new_key = key.replace("LayerNorm.gamma", "LayerNorm.weight")
                if key.endswith("LayerNorm.beta"):  # compatibility with v0.5 models
                    new_key = key.replace("LayerNorm.beta", "LayerNorm.bias")
                if new_key is not None:
                    state_dict[new_key] = state_dict.pop(key)

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
        log_class_usage(__class__)

    def _encoder(self, input_tuple: Tuple[torch.Tensor, ...]):
        tokens, pad_mask, segment_labels, _ = input_tuple
        last_encoded_layers, pooled_output, encoded_layers = self.bert(
            tokens,
            attention_mask=pad_mask,
            token_type_ids=segment_labels,
            output_hidden_states=True,
        )
        return encoded_layers, pooled_output

    def _embedding(self):
        # used to tie weights in MaskedLM model
        return self.bert.embeddings.word_embeddings
