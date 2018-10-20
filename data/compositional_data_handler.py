#!/usr/bin/env python3
from typing import Any, Dict, List

from pytext.common.constants import DatasetFieldName, DFColumn
from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig
from pytext.data.featurizer import InputRecord
from pytext.fields import ActionField, DictFeatureField, Field, TextFeatureField

from .data_handler import DataHandler


TREE_COLUMN = "tree"
ACTION_FEATURE_FIELD = "action_idx_feature"
ACTION_LABEL_FIELD = "action_idx_label"


class CompositionalDataHandler(DataHandler):
    class Config(ConfigBase, DataHandler.Config):
        columns_to_read: List[str] = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
        ]

    FULL_FEATURES = [
        DatasetFieldName.TEXT_FIELD,
        DatasetFieldName.DICT_FIELD,
        ACTION_FEATURE_FIELD,
    ]

    @classmethod
    def from_config(cls, config: Config, feature_config: FeatureConfig, **kwargs):
        word_feat_config = feature_config.word_feat
        features: Dict[str, Field] = {
            # TODO assuming replacing numbers with NUM and unkify be done in featurizer
            DatasetFieldName.TEXT_FIELD: TextFeatureField(
                pretrained_embeddings_path=word_feat_config.pretrained_embeddings_path,
                embed_dim=word_feat_config.embed_dim,
                embedding_init_strategy=word_feat_config.embedding_init_strategy,
                vocab_file=word_feat_config.vocab_file,
                vocab_size=word_feat_config.vocab_size,
                vocab_from_train_data=word_feat_config.vocab_from_train_data,
            )
        }
        if feature_config.dict_feat and feature_config.dict_feat.embed_dim > 0:
            features[DatasetFieldName.DICT_FIELD] = DictFeatureField()
        features[ACTION_FEATURE_FIELD] = ActionField()
        return cls(
            raw_columns=config.columns_to_read,
            features=features,
            labels={ACTION_LABEL_FIELD: ActionField()},
            train_path=config.train_path,
            eval_path=config.eval_path,
            test_path=config.test_path,
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            test_batch_size=config.test_batch_size,
            **kwargs
        )

    def _input_from_batch(self, batch, is_train):
        # text_input[0] is contains numericalized tokens.
        text_input = getattr(batch, DatasetFieldName.TEXT_FIELD)
        return (text_input[0], text_input[1]) + tuple(
            getattr(batch, name, None)
            for name in self.FULL_FEATURES
            if name != DatasetFieldName.TEXT_FIELD
        )

    def preprocess_row(self, row_data: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # TODO avoid this mapping
        raw_input = InputRecord(
            raw_text=row_data[DFColumn.UTTERANCE],
            raw_gazetteer_feats=row_data[DFColumn.DICT_FEAT],
        )

        features = self.featurizer.featurize(raw_input)

        # TODO should implement a version whose params are plain intent, slot and
        # get rid of IntentFrame class
        # actions = intent_frame_to_tree(
        #     dict_to_thrift(
        #         IntentFrame,
        #         {
        #             "utterance": row_data[DFColumn.UTTERANCE],
        #             "intent": row_data[DFColumn.DOC_LABEL],
        #             "slots": row_data[DFColumn.WORD_LABEL],
        #         },
        #     )
        # ).to_actions()
        return {
            # TODO set_tokens_indices, should implement another field
            # TODO is it the same with the original tokens seq?
            DatasetFieldName.TEXT_FIELD: features.tokens,
            DatasetFieldName.DICT_FIELD: (
                features.gazetteer_feats,
                features.gazetteer_feat_weights,
                features.gazetteer_feat_lengths,
            ),
            # ACTION_FEATURE_FIELD: actions,
            # ACTION_LABEL_FIELD: actions,
        }
