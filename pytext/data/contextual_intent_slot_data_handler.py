#!/usr/bin/env python3

from typing import Any, Dict, List

from pytext.config.contextual_intent_slot import (
    ExtraField,
    ModelInput,
    ModelInputConfig,
    Target,
    TargetConfig,
)
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data.featurizer import InputRecord
from pytext.fields import (
    CharFeatureField,
    DictFeatureField,
    DocLabelField,
    Field,
    FloatField,
    PretrainedModelEmbeddingField,
    RawField,
    SeqFeatureField,
    TextFeatureField,
    WordLabelField,
    create_fields,
)
from pytext.utils import data_utils

from .joint_data_handler import JointModelDataHandler


class RawData:
    DOC_LABEL = "doc_label"
    WORD_LABEL = "word_label"
    TEXT = "text"
    DICT_FEAT = "dict_feat"
    DOC_WEIGHT = "doc_weight"
    WORD_WEIGHT = "word_weight"


class ContextualIntentSlotModelDataHandler(JointModelDataHandler):
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
        features: Dict[str, Field] = create_fields(
            feature_config,
            {
                ModelInput.TEXT: TextFeatureField,
                ModelInput.DICT: DictFeatureField,
                ModelInput.CHAR: CharFeatureField,
                ModelInput.PRETRAINED: PretrainedModelEmbeddingField,
                ModelInput.SEQ: SeqFeatureField,
            },
        )

        labels: Dict[str, Field] = create_fields(
            target_config,
            {Target.DOC_LABEL: DocLabelField, Target.WORD_LABEL: WordLabelField},
        )

        extra_fields: Dict[str, Field] = {
            ExtraField.DOC_WEIGHT: FloatField(),
            ExtraField.WORD_WEIGHT: FloatField(),
            ExtraField.RAW_WORD_LABEL: RawField(),
            ExtraField.TOKEN_RANGE: RawField(),
            ExtraField.INDEX_FIELD: RawField(),
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

    def preprocess_row(self, row_data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        sequence = data_utils.parse_json_array(row_data[RawData.TEXT])

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
            ModelInput.PRETRAINED: features_list[-1].pretrained_token_embedding,
            # labels
            Target.DOC_LABEL: row_data[RawData.DOC_LABEL],
            # extra data
            # TODO move the logic to FloatField
            ExtraField.DOC_WEIGHT: row_data.get(RawData.DOC_WEIGHT) or 1.0,
            ExtraField.WORD_WEIGHT: row_data.get(RawData.WORD_WEIGHT) or 1.0,
            ExtraField.RAW_WORD_LABEL: row_data[RawData.WORD_LABEL],
            ExtraField.INDEX_FIELD: idx,
            ExtraField.UTTERANCE: row_data[RawData.TEXT],
            ExtraField.TOKEN_RANGE: features_list[-1].token_ranges,
        }
        if Target.WORD_LABEL in self.labels:
            # TODO move it into word label field
            res[Target.WORD_LABEL] = data_utils.align_slot_labels(
                features_list[-1].token_ranges,
                row_data[RawData.WORD_LABEL],
                self.labels[Target.WORD_LABEL].use_bio_labels,
            )
        return res

    def _train_input_from_batch(self, batch):
        text_input = getattr(batch, ModelInput.TEXT)
        seq_input = getattr(batch, ModelInput.SEQ)
        return (
            # text_input[0] contains the word embeddings,
            # text_input[1] contains the lengths of each word
            text_input[0],
            *(
                getattr(batch, key)
                for key in self.features
                if key not in [ModelInput.TEXT, ModelInput.SEQ]
            ),
            seq_input[0],
            text_input[1],
            seq_input[1],
        )
