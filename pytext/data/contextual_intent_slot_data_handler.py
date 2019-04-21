#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List

from pytext.config.contextual_intent_slot import (
    ExtraField,
    ModelInput,
    ModelInputConfig,
    TargetConfig,
)
from pytext.config.field_config import DocLabelConfig, WordLabelConfig
from pytext.data.featurizer import InputRecord
from pytext.fields import (
    CharFeatureField,
    ContextualTokenEmbeddingField,
    DictFeatureField,
    DocLabelField,
    Field,
    FloatField,
    FloatVectorField,
    RawField,
    SeqFeatureField,
    TextFeatureField,
    WordLabelField,
    create_fields,
    create_label_fields,
)
from pytext.utils import data

from .joint_data_handler import JointModelDataHandler


class RawData:
    DOC_LABEL = "doc_label"
    WORD_LABEL = "word_label"
    TEXT = "text"
    DICT_FEAT = "dict_feat"
    DOC_WEIGHT = "doc_weight"
    WORD_WEIGHT = "word_weight"
    DENSE_FEAT = "dense_feat"


class ContextualIntentSlotModelDataHandler(JointModelDataHandler):
    """
    Data Handler to build pipeline to process data and generate tensors to be consumed
    by ContextualIntentSlotModel. Columns of Input data includes:

        1. doc label for intent classification
        2. word label for slot tagging of the last utterance
        3. a sequence of utterances (e.g., a dialog)
        4. Optional dictionary feature contained in the last utterance
        5. Optional doc weight that stands for the weight of intent task in joint loss.
        6. Optional word weight that stands for the weight of slot task in joint loss.

    Attributes:
        raw_columns: columns to read from data source. In case of files, the order
            should match the data stored in that file. Raw columns include
            ::

                [
                    RawData.DOC_LABEL,
                    RawData.WORD_LABEL,
                    RawData.TEXT,
                    RawData.DICT_FEAT (Optional),
                    RawData.DOC_WEIGHT (Optional),
                    RawData.WORD_WEIGHT (Optional),
                ]

        labels: doc labels and word labels
        features: embeddings generated from sequences of utterances and
            dictionary features of the last utterance
        extra_fields: doc weights, word weights, and etc.
    """

    class Config(JointModelDataHandler.Config):
        columns_to_read: List[str] = [
            RawData.DOC_LABEL,
            RawData.WORD_LABEL,
            RawData.TEXT,
            RawData.DICT_FEAT,
            RawData.DOC_WEIGHT,
            RawData.WORD_WEIGHT,
        ]

    @classmethod
    def from_config(
        cls,
        config: Config,
        feature_config: ModelInputConfig,
        target_config: TargetConfig,
        **kwargs,
    ):
        """Factory method to construct an instance of
        ContextualIntentSlotModelDataHandler object from the module's config,
        model input config and target config.

        Args:
            config (Config): Configuration object specifying all the
                parameters of ContextualIntentSlotModelDataHandler.
            feature_config (ModelInputConfig): Configuration object specifying
                model input.
            target_config (TargetConfig): Configuration object specifying target.

        Returns:
            type: An instance of ContextualIntentSlotModelDataHandler.

        """
        features: Dict[str, Field] = create_fields(
            feature_config,
            {
                ModelInput.TEXT: TextFeatureField,
                ModelInput.DICT: DictFeatureField,
                ModelInput.CHAR: CharFeatureField,
                ModelInput.CONTEXTUAL_TOKEN_EMBEDDING: ContextualTokenEmbeddingField,
                ModelInput.SEQ: SeqFeatureField,
                ModelInput.DENSE: FloatVectorField,
            },
        )

        # Label fields.
        labels: Dict[str, Field] = create_label_fields(
            target_config,
            {
                DocLabelConfig._name: DocLabelField,
                WordLabelConfig._name: WordLabelField,
            },
        )

        extra_fields: Dict[str, Field] = {
            ExtraField.DOC_WEIGHT: FloatField(),
            ExtraField.WORD_WEIGHT: FloatField(),
            ExtraField.RAW_WORD_LABEL: RawField(),
            ExtraField.TOKEN_RANGE: RawField(),
            ExtraField.UTTERANCE: RawField(),
        }

        kwargs.update(config.items())
        return cls(
            raw_columns=config.columns_to_read,
            labels=labels,
            features=features,
            extra_fields=extra_fields,
            **kwargs,
        )

    def preprocess_row(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess steps for a single input row: 1. apply tokenization to a
        sequence of utterances; 2. process dictionary features to align with
        the last utterance. 3. align word labels with the last utterance.

        Args:
            row_data (Dict[str, Any]): Dict of one row data with column names as keys.
                Keys includes "doc_label", "word_label", "text", "dict_feat",
                "word weight" and "doc weight".

        Returns:
            Dict[str, Any]: Preprocessed dict of one row data includes:

                "seq_word_feat" (list of list of string)
                    tokenized words of sequence of utterances
                "word_feat" (list of string)
                    tokenized words of last utterance
                "raw_word_label" (string)
                    raw word label
                "token_range" (list of tuple)
                    token ranges of word labels, each tuple contains the start
                    position index and the end position index
                "utterance" (list of string)
                    raw utterances
                "word_label" (list of string)
                    list of labels of words in last utterance
                "doc_label" (string)
                    doc label for intent classification
                "word_weight" (float)
                    weight of word label
                "doc_weight" (float)
                    weight of document label
                "dict_feat" (tuple, optional)
                    tuple of three lists, the first is the label of each words,
                    the second is the weight of the feature, the third is the
                    length of the feature.

        """
        sequence = data.parse_json_array(row_data[RawData.TEXT])

        # ignore dictionary feature for context sentences other than the last one
        features_list = [
            self.featurizer.featurize(InputRecord(raw_text=utterance))
            for utterance in sequence[:-1]
        ]

        # adding dictionary feature for the last (current) message
        features_list.append(
            self.featurizer.featurize(
                InputRecord(
                    raw_text=sequence[-1],
                    raw_gazetteer_feats=row_data.get(ModelInput.DICT, ""),
                )
            )
        )

        res = {
            # features
            ModelInput.SEQ: [utterance.tokens for utterance in features_list],
            ModelInput.TEXT: features_list[-1].tokens,
            ModelInput.DICT: (
                features_list[-1].gazetteer_feats,
                features_list[-1].gazetteer_feat_weights,
                features_list[-1].gazetteer_feat_lengths,
            ),
            ModelInput.CHAR: features_list[-1].characters,
            ModelInput.CONTEXTUAL_TOKEN_EMBEDDING: features_list[
                -1
            ].contextual_token_embedding,
            # labels
            DocLabelConfig._name: row_data[RawData.DOC_LABEL],
            # extra data
            # TODO move the logic to FloatField
            ExtraField.DOC_WEIGHT: row_data.get(RawData.DOC_WEIGHT) or 1.0,
            ExtraField.WORD_WEIGHT: row_data.get(RawData.WORD_WEIGHT) or 1.0,
            ExtraField.RAW_WORD_LABEL: row_data[RawData.WORD_LABEL],
            ExtraField.UTTERANCE: row_data[RawData.TEXT],
            ExtraField.TOKEN_RANGE: features_list[-1].token_ranges,
        }

        if RawData.DENSE_FEAT in row_data:
            res[ModelInput.DENSE] = row_data.get(RawData.DENSE_FEAT)

        if WordLabelConfig._name in self.labels:
            # TODO move it into word label field
            res[WordLabelConfig._name] = data.align_slot_labels(
                features_list[-1].token_ranges,
                row_data[RawData.WORD_LABEL],
                self.labels[WordLabelConfig._name].use_bio_labels,
            )
        return res

    def _train_input_from_batch(self, batch):
        text_input = getattr(batch, ModelInput.TEXT)
        seq_input = getattr(batch, ModelInput.SEQ)
        result = (
            # text_input[0] contains the word embeddings,
            # text_input[1] contains the lengths of each word
            text_input[0],
            *(
                getattr(batch, key)
                for key in self.features
                if key not in [ModelInput.TEXT, ModelInput.SEQ, ModelInput.DENSE]
            ),
            seq_input[0],
            text_input[1],
            seq_input[1],
        )
        # Append dense faeture to decoder layer at the end.
        if ModelInput.DENSE in self.features:
            result = result + (getattr(batch, ModelInput.DENSE),)
        return result
