#!/usr/bin/env python3

from typing import Any, Dict, List

import pandas as pd
from pytext.common.constants import DatasetFieldName, DFColumn
from pytext.common.registry import DATA_HANDLER, component
from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data.shared_featurizer import SharedFeaturizer
from pytext.utils import data_utils

from .data_handler import DataHandler
from .field import (
    CapFeature,
    CharFeature,
    DictFeature,
    DocLabelField,
    Field,
    LossWeightField,
    RawField,
    TextFeature,
    WordLabelField,
)


SEQ_LENS = "seq_lens"


class JointTextModelDataHandlerConfig(ConfigBase):
    columns_to_read: List[str] = [
        DFColumn.DOC_LABEL,
        DFColumn.WORD_LABEL,
        DFColumn.UTTERANCE,
        DFColumn.DICT_FEAT,
        DFColumn.DOC_WEIGHT,
        DFColumn.WORD_WEIGHT,
    ]
    preprocess_workers: int = 32
    pretrained_embeds_file: str = ""


@component(DATA_HANDLER, config_cls=JointTextModelDataHandlerConfig)
class JointModelDataHandler(DataHandler):

    FULL_FEATURES = [
        DatasetFieldName.TEXT_FIELD,
        DatasetFieldName.DICT_FIELD,
        DatasetFieldName.CAP_FIELD,
        DatasetFieldName.CHAR_FIELD,
    ]

    @classmethod
    def from_config(
        cls,
        config: JointTextModelDataHandlerConfig,
        feature_config: FeatureConfig,
        label_config: LabelConfig,
        **kwargs
    ):
        features: List[Field] = [
            TextFeature(
                DatasetFieldName.TEXT_FIELD,
                export_input_names=feature_config.word_feat.export_input_names,
            )
        ]
        if feature_config.dict_feat:
            features.append(
                DictFeature(
                    DatasetFieldName.DICT_FIELD,
                    export_input_names=feature_config.dict_feat.export_input_names,
                )
            )
        if feature_config.cap_feat:
            features.append(CapFeature(DatasetFieldName.CAP_FIELD))
        if feature_config.char_feat:
            features.append(CharFeature(DatasetFieldName.CHAR_FIELD))

        labels: List[Field] = []
        if label_config.doc_label:
            labels.append(DocLabelField(DatasetFieldName.DOC_LABEL_FIELD))
        if label_config.word_label:
            labels.append(
                WordLabelField(
                    DatasetFieldName.WORD_LABEL_FIELD,
                    use_bio_labels=label_config.word_label.use_bio_labels,
                )
            )
        extra_fields: List[Field] = [
            LossWeightField(DatasetFieldName.DOC_WEIGHT_FIELD),
            LossWeightField(DatasetFieldName.WORD_WEIGHT_FIELD),
            RawField(DatasetFieldName.RAW_WORD_LABEL),
            RawField(DatasetFieldName.TOKEN_RANGE_PAIR),
            RawField(DatasetFieldName.INDEX_FIELD),
        ]

        return cls(
            raw_columns=config.columns_to_read,
            labels=labels,
            features=features,
            extra_fields=extra_fields,
            featurizer=SharedFeaturizer(),
            num_workers=config.preprocess_workers,
            pretrained_embeds_file=config.pretrained_embeds_file,
            embed_dim=feature_config.word_feat.embed_dim,
            embed_init_strategy=feature_config.word_feat.embed_init_strategy,
        )

    def __init__(
        self, featurizer: SharedFeaturizer, num_workers: int, **kwargs
    ) -> None:

        super().__init__(**kwargs)
        # configs
        self.featurizer = featurizer
        self.num_workers = num_workers

        self.df_to_feat_func_map = {
            # features
            DatasetFieldName.TEXT_FIELD: lambda row, field: row[
                DFColumn.MODEL_FEATS
            ].tokens,
            DatasetFieldName.DICT_FIELD: lambda row, field: (
                row[DFColumn.MODEL_FEATS].dictFeats,
                row[DFColumn.MODEL_FEATS].dictFeatWeights,
                row[DFColumn.MODEL_FEATS].dictFeatLengths,
            ),
            DatasetFieldName.CHAR_FIELD: lambda row, field: row[
                DFColumn.MODEL_FEATS
            ].chars,
            DatasetFieldName.CAP_FIELD: lambda row, field: [
                data_utils.capitalization_feature(t)
                for (t, (_, __)) in row[DFColumn.TOKEN_RANGE_PAIR]
            ],
            # labels
            DatasetFieldName.DOC_LABEL_FIELD: DFColumn.DOC_LABEL,
            DatasetFieldName.WORD_LABEL_FIELD: lambda row, field: data_utils.align_slot_labels(
                row[DFColumn.TOKEN_RANGE_PAIR],
                row[DFColumn.WORD_LABEL],
                field.use_bio_labels,
            ),
            # extra context
            DatasetFieldName.DOC_WEIGHT_FIELD: lambda row, field: row.get(
                DFColumn.DOC_WEIGHT
            )
            or 1.0,
            DatasetFieldName.WORD_WEIGHT_FIELD: lambda row, field: row.get(
                DFColumn.WORD_WEIGHT
            )
            or 1.0,
            DatasetFieldName.RAW_WORD_LABEL: DFColumn.WORD_LABEL,
            DatasetFieldName.INDEX_FIELD: self.DF_INDEX,
        }

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if DFColumn.DICT_FEAT not in df:
            df[DFColumn.DICT_FEAT] = ""

        df[DFColumn.RAW_FEATS] = df.apply(
            lambda row: (row[DFColumn.UTTERANCE], row[DFColumn.DICT_FEAT]), axis=1
        )

        df[DFColumn.MODEL_FEATS] = pd.Series(
            self.featurizer.featurize_parallel(
                df[DFColumn.RAW_FEATS].tolist(), self.num_workers
            )
        )

        df[DFColumn.TOKEN_RANGE_PAIR] = [
            data_utils.parse_token(
                row[DFColumn.UTTERANCE], row[DFColumn.MODEL_FEATS].tokenRanges
            )
            for _, row in df.iterrows()
        ]

        return df

    def _input_from_batch(self, batch):
        # batch.text[1] is the length of each word
        return (batch.text[0], batch.text[1]) + tuple(
            getattr(batch, name, None)
            for name in self.FULL_FEATURES
            if name != DatasetFieldName.TEXT_FIELD
        )

    def _context_from_batch(self, batch):
        # batch.text[1] is the length of each word
        res = {SEQ_LENS: batch.text[1]}
        res.update(super()._context_from_batch(batch))
        return res

    def _gen_extra_metadata(self) -> Dict[str, Any]:
        return {
            "class_names": [label.field.vocab.itos for label in self.labels],
            "feature_itos_map": {
                k: v.itos for k, v in self.metadata["feature_vocabs"].items()
            },
        }
