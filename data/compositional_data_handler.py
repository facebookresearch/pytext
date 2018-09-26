#!/usr/bin/env python3

from typing import List

import pandas as pd
from eit.llama.common.thriftutils import dict_to_thrift
from messenger.assistant.cu.core.ttypes import IntentFrame
from pytext.common.constants import DatasetFieldName, DFColumn
from pytext.config import ConfigBase
from pytext.fb.data.assistant_featurizer import (
    AssistantFeaturizer,
    parse_assistant_raw_record,
)
from pytext.fb.rnng.tools.annotation_to_intent_frame import intent_frame_to_tree
from pytext.fields import ActionField, DictFeatureField, TextFeatureField
from pytext.fields.utils import reverse_tensor
from pytext.utils import data_utils

from .data_handler import DataHandler


TREE_COLUMN = "tree"


class CompositionalDataHandler(DataHandler):
    class Config(ConfigBase, DataHandler.Config):
        columns_to_read: List[str] = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
        ]

    def __init__(self, **kwargs) -> None:
        super().__init__(
            features={
                # TODO assuming replacing numbers with NUM and unkify be done in featurizer
                DatasetFieldName.TEXT_FIELD: TextFeatureField(
                    postprocessing=reverse_tensor
                ),
                DatasetFieldName.DICT_FIELD: DictFeatureField(),
                "action_idx_feature": ActionField(),
            },
            labels={"action_idx_label": ActionField()},
            **kwargs
        )
        self.featurizer = AssistantFeaturizer()

        self.df_to_example_func_map = {
            # TODO set_tokens_indices, should implement another field
            # TODO is it the same with the original tokens seq?
            DatasetFieldName.TEXT_FIELD: lambda row, field: row[
                DFColumn.MODEL_FEATS
            ].tokens,
            DatasetFieldName.DICT_FIELD: lambda row, field: (
                row[DFColumn.MODEL_FEATS].dictFeats,
                row[DFColumn.MODEL_FEATS].dictFeatWeights,
                row[DFColumn.MODEL_FEATS].dictFeatLengths,
            ),
            "action_idx_feature": lambda row, field: row[TREE_COLUMN].to_actions(),
            "action_idx_label": lambda row, field: row[TREE_COLUMN].to_actions(),
        }

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if DFColumn.DICT_FEAT not in df:
            df[DFColumn.DICT_FEAT] = ""

        df[DFColumn.RAW_FEATS] = df.apply(
            lambda row: parse_assistant_raw_record(
                row[DFColumn.UTTERANCE],
                row[DFColumn.DICT_FEAT]
            ),
            axis=1
        )

        df[DFColumn.MODEL_FEATS] = pd.Series(
            self.featurizer.featurize_batch(
                df[DFColumn.RAW_FEATS].tolist()
            )
        )

        df[DFColumn.TOKEN_RANGE_PAIR] = [
            data_utils.parse_token(
                row[DFColumn.UTTERANCE], row[DFColumn.MODEL_FEATS].tokenRanges
            )
            for _, row in df.iterrows()
        ]

        # TODO should implement a version whose params are plain intent, slot and
        # get rid of IntentFrame class
        df[TREE_COLUMN] = df.apply(
            lambda row: intent_frame_to_tree(
                dict_to_thrift(
                    IntentFrame,
                    {
                        "utterance": row[DFColumn.UTTERANCE],
                        "intent": row[DFColumn.DOC_LABEL],
                        "slots": row[DFColumn.WORD_LABEL],
                    },
                )
            ),
            axis=1,
        )
        return df
