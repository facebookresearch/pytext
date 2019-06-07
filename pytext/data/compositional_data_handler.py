#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Any, Dict, List

from pytext.common.constants import DatasetFieldName, DFColumn
from pytext.config.field_config import FeatureConfig
from pytext.data.data_handler import DataHandler
from pytext.data.data_structures.annotation import (
    REDUCE,
    SHIFT,
    Annotation,
    is_intent_nonterminal,
    is_slot_nonterminal,
    is_unsupported,
    is_valid_nonterminal,
)
from pytext.data.featurizer import InputRecord
from pytext.fields import (
    ActionField,
    ContextualTokenEmbeddingField,
    DictFeatureField,
    Field,
    RawField,
    TextFeatureFieldWithSpecialUnk,
)


TREE_COLUMN = "tree"
ACTION_FEATURE_FIELD = "action_idx_feature"
ACTION_LABEL_FIELD = "action_idx_label"


class CompositionalDataHandler(DataHandler):
    class Config(DataHandler.Config):
        columns_to_read: List[str] = [
            DFColumn.DOC_LABEL,
            DFColumn.WORD_LABEL,
            DFColumn.UTTERANCE,
            DFColumn.DICT_FEAT,
            DFColumn.SEQLOGICAL,
        ]
        train_batch_size: int = 1
        eval_batch_size: int = 1
        test_batch_size: int = 1

    FULL_FEATURES = [
        DatasetFieldName.TEXT_FIELD,
        DatasetFieldName.DICT_FIELD,
        ACTION_FEATURE_FIELD,
        DatasetFieldName.CONTEXTUAL_TOKEN_EMBEDDING,
    ]

    @classmethod
    def from_config(
        cls, config: Config, feature_config: FeatureConfig, *args, **kwargs
    ):
        word_feat_config = feature_config.word_feat
        features: Dict[str, Field] = {
            DatasetFieldName.TEXT_FIELD: TextFeatureFieldWithSpecialUnk(
                pretrained_embeddings_path=word_feat_config.pretrained_embeddings_path,
                embed_dim=word_feat_config.embed_dim,
                embedding_init_strategy=word_feat_config.embedding_init_strategy,
                vocab_file=word_feat_config.vocab_file,
                vocab_size=word_feat_config.vocab_size,
                vocab_from_train_data=word_feat_config.vocab_from_train_data,
                vocab_from_all_data=word_feat_config.vocab_from_all_data,
                min_freq=word_feat_config.min_freq,
                pad_token=None,
            )
        }
        if feature_config.dict_feat and feature_config.dict_feat.embed_dim > 0:
            features[DatasetFieldName.DICT_FIELD] = DictFeatureField()

        # Adding action_field to list of features so that it can be passed to
        # RNNGParser's forward method during training time.
        action_field = ActionField()  # Use the same field for label too.
        features[ACTION_FEATURE_FIELD] = action_field

        if feature_config.contextual_token_embedding:
            features[
                DatasetFieldName.CONTEXTUAL_TOKEN_EMBEDDING
            ] = ContextualTokenEmbeddingField(
                embed_dim=feature_config.contextual_token_embedding.embed_dim
            )

        extra_fields: Dict[str, Field] = {
            DatasetFieldName.TOKENS: RawField(),
            DatasetFieldName.UTTERANCE_FIELD: RawField(),
        }

        return cls(
            raw_columns=config.columns_to_read,
            features=features,
            labels={ACTION_LABEL_FIELD: action_field},
            extra_fields=extra_fields,
            train_path=config.train_path,
            eval_path=config.eval_path,
            test_path=config.test_path,
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            test_batch_size=config.test_batch_size,
            shuffle=config.shuffle,
            sort_within_batch=config.sort_within_batch,
            column_mapping=config.column_mapping,
            **kwargs,
        )

    def _gen_extra_metadata(self):
        self.metadata.actions_vocab = self.features[ACTION_FEATURE_FIELD].vocab
        actions_vocab_dict: Dict = self.features[ACTION_FEATURE_FIELD].vocab.stoi

        # SHIFT and REDUCE indices.
        self.metadata.shift_idx: int = actions_vocab_dict[SHIFT]
        self.metadata.reduce_idx: int = actions_vocab_dict[REDUCE]

        # unsupported instances
        self.metadata.ignore_subNTs_roots: List[int] = [
            actions_vocab_dict[nt]
            for nt in actions_vocab_dict.keys()
            if is_unsupported(nt)
        ]
        self.metadata.valid_NT_idxs: List[int] = [
            actions_vocab_dict[nt]
            for nt in actions_vocab_dict.keys()
            if is_valid_nonterminal(nt)
        ]
        self.metadata.valid_IN_idxs: List[int] = [
            actions_vocab_dict[nt]
            for nt in actions_vocab_dict.keys()
            if is_intent_nonterminal(nt)
        ]
        self.metadata.valid_SL_idxs: List[int] = [
            actions_vocab_dict[nt]
            for nt in actions_vocab_dict.keys()
            if is_slot_nonterminal(nt)
        ]

    def _train_input_from_batch(self, batch):
        # text_input[0] is contains numericalized tokens.
        text_input = getattr(batch, DatasetFieldName.TEXT_FIELD)
        m_inputs = [text_input[0], text_input[1]]
        for name in self.FULL_FEATURES:
            if name == DatasetFieldName.TEXT_FIELD:
                continue
            input = getattr(batch, name, None)
            if name == ACTION_FEATURE_FIELD:
                input = input.tolist()  # Action needn't be passed as Tensor obj.
            m_inputs.append(input)
        # beam size and topk
        m_inputs.extend([1, 1])
        return m_inputs

    def _test_input_from_batch(self, batch):
        text_input = getattr(batch, DatasetFieldName.TEXT_FIELD)
        return [
            text_input[0],
            text_input[1],
            getattr(batch, DatasetFieldName.DICT_FIELD, None),
            [],
            getattr(batch, DatasetFieldName.CONTEXTUAL_TOKEN_EMBEDDING, None),
            1,
            1,
        ]

    def preprocess_row(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        utterance = row_data.get(DFColumn.UTTERANCE, "")
        features = self.featurizer.featurize(
            InputRecord(
                raw_text=utterance,
                raw_gazetteer_feats=row_data.get(DFColumn.DICT_FEAT, ""),
            )
        )
        actions = ""
        # training time
        if DFColumn.SEQLOGICAL in row_data:
            annotation = Annotation(row_data[DFColumn.SEQLOGICAL], utterance)
            actions = annotation.tree.to_actions()

            # Seqlogical format is required for building the tree representation of
            # compositional utterances and, it depends on tokenization.
            # Here during preprocessing, if the tokens produced from Featurizer
            # and those from the seqlogical format are not consistent, then it leads
            # to inconsistent non terminals and actions which in turn leads to
            # the model's forward method throwing an exception.
            # This should NOT happen but the check below is to make sure the
            # model training doesn't fail just in case there's inconsistency.
            tokens_from_seqlogical = annotation.tree.list_tokens()
            try:
                assert len(features.tokens) == len(tokens_from_seqlogical)
                for t1, t2 in zip(features.tokens, tokens_from_seqlogical):
                    assert t1.lower() == t2.lower()
            except AssertionError:
                print(
                    "\nTokens from Featurizer and Seqlogical format are not same "
                    + f'for the utterance "{utterance}"'
                )
                print(
                    f"{len(features.tokens)} tokens from Featurizer: {features.tokens}"
                )
                print(
                    f"{len(tokens_from_seqlogical)} tokens from Seqlogical format: "
                    + f"{tokens_from_seqlogical}"
                )
                return {}

        contextual_token_embedding = 0
        if (
            features.contextual_token_embedding
            and len(features.contextual_token_embedding) > 0
        ):
            contextual_token_embedding = features.contextual_token_embedding

        return {
            DatasetFieldName.TEXT_FIELD: features.tokens,
            DatasetFieldName.DICT_FIELD: (
                features.gazetteer_feats,
                features.gazetteer_feat_weights,
                features.gazetteer_feat_lengths,
            ),
            ACTION_FEATURE_FIELD: actions,
            ACTION_LABEL_FIELD: copy.deepcopy(actions),
            DatasetFieldName.TOKENS: features.tokens,
            DatasetFieldName.UTTERANCE_FIELD: utterance,
            DatasetFieldName.CONTEXTUAL_TOKEN_EMBEDDING: contextual_token_embedding,
        }
