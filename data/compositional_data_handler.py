#!/usr/bin/env python3

from typing import List

import pandas as pd
from pytext.common.constants import DatasetFieldName, DFColumn
from pytext.common.registry import DATA_HANDLER, component
from pytext.config import ConfigBase
from pytext.data.field import DictFeature, Field, TextFeature
from pytext.data.shared_featurizer import SharedFeaturizer
from pytext.rnng.tools.annotation_to_intent_frame import intent_frame_to_tree
from pytext.utils import data_utils
from eit.llama.common.thriftutils import dict_to_thrift
from messenger.assistant.cu.core.ttypes import IntentFrame
from torchtext import data as textdata

from .data_handler import DataHandler


# TODO move it to field folder
def _reverse(arr, vocab):
    return [ex[::-1] for ex in arr]


# TODO move it to field folder
class ActionField(Field):
    def __init__(self, name):
        super().__init__(name)
        self.field = textdata.Field(
            use_vocab=True,
            sequential=True,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
            unk_token=None,  # Don't include unk in the list of labels
            # reverse the tensor
            postprocessing=_reverse,
        )


TREE_COLUMN = "tree"


class CompositionalDataHandlerConfig(ConfigBase):
    columns_to_read: List[str] = [
        DFColumn.DOC_LABEL,
        DFColumn.WORD_LABEL,
        DFColumn.UTTERANCE,
        DFColumn.DICT_FEAT,
    ]
    preprocess_workers: int = 32
    pretrained_embeds_file: str = ""
    shuffle: bool = True


@component(DATA_HANDLER, config_cls=CompositionalDataHandlerConfig)
class CompositionalDataHandler(DataHandler):
    def __init__(self, num_workers=1, **kwargs) -> None:
        super().__init__(
            features=[
                # TODO assuming replacing numbers with NUM and unkify be done in featurizer
                TextFeature(DatasetFieldName.TEXT_FIELD, postprocessing=_reverse),
                DictFeature(DatasetFieldName.DICT_FIELD),
                ActionField("action_idx_feature"),
            ],
            labels=[ActionField("action_idx_label")],
            **kwargs
        )
        self.featurizer = SharedFeaturizer()
        self.num_workers = num_workers

        self.df_to_feat_func_map = {
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
