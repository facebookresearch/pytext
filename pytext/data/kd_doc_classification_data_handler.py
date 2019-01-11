#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List

from pytext.config.field_config import DocLabelConfig
from pytext.config.kd_doc_classification import (
    ExtraField,
    ModelInput,
    ModelInputConfig,
    Target,
    TargetConfig,
)
from pytext.data.featurizer import InputRecord
from pytext.fields import (
    CharFeatureField,
    DictFeatureField,
    DocLabelField,
    Field,
    PretrainedModelEmbeddingField,
    RawField,
    TextFeatureField,
    create_fields,
    create_label_fields,
)
from pytext.utils.data_utils import parse_json_array
from pytext.utils.python_utils import cls_vars
from torchtext import data as textdata

from .doc_classification_data_handler import DocClassificationDataHandler


class RawData:
    DOC_LABEL = "doc_label"
    TEXT = "text"
    DICT_FEAT = "dict_feat"
    TARGET_PROBS = "target_probs"
    TARGET_LOGITS = "target_logits"
    TARGET_LABELS = "target_labels"


class KDDocClassificationDataHandler(DocClassificationDataHandler):
    """
    The `KDDocClassificationDataHandler` prepares the data for knowledge distillation
    document classififcation. Each sentence is read line by line with its label as the
    target.
    """

    class Config(DocClassificationDataHandler.Config):
        """
        Configuration class for `KDDocClassificationDataHandler`.

        Attributes:
            columns_to_read (List[str]): List containing the names of the
                columns to read from the data files.
            max_seq_len (int): Maximum sequence length for the input. The input
                is trimmed after the maximum sequence length.
        """

        columns_to_read: List[str] = cls_vars(RawData)
        max_seq_len: int = -1

    @classmethod
    def from_config(
        cls,
        config: Config,
        model_input_config: ModelInputConfig,
        target_config: TargetConfig,
        **kwargs,
    ):
        """
        Factory method to construct an instance of `DocClassificationDataHandler`
        from the module's config object and feature config object.

        Args:
            config (DocClassificationDataHandler.Config): Configuration object
                specifying all the parameters of `DocClassificationDataHandler`.
            model_input_config (ModelInputConfig): Configuration object
                specifying all the parameters of the model config.
            target_config (TargetConfig): Configuration object specifying all
                the parameters of the target.

        Returns:
            type: An instance of `KDDocClassificationDataHandler`.
        """
        model_input_fields: Dict[str, Field] = create_fields(
            model_input_config,
            {
                ModelInput.WORD_FEAT: TextFeatureField,
                ModelInput.DICT_FEAT: DictFeatureField,
                ModelInput.CHAR_FEAT: CharFeatureField,
                ModelInput.PRETRAINED_MODEL_EMBEDDING: PretrainedModelEmbeddingField,
            },
        )
        target_fields: Dict[str, Field] = create_label_fields(
            target_config, {DocLabelConfig._name: DocLabelField}
        )
        extra_fields: Dict[str, Field] = {ExtraField.RAW_TEXT: RawField()}
        if target_config.target_prob:
            target_fields[Target.TARGET_PROB_FIELD] = RawField()
            target_fields[Target.TARGET_LOGITS_FIELD] = RawField()

        if target_config.target_prob:
            extra_fields[Target.TARGET_LABEL_FIELD] = RawField()
        kwargs.update(config.items())
        return cls(
            raw_columns=config.columns_to_read,
            labels=target_fields,
            features=model_input_fields,
            extra_fields=extra_fields,
            **kwargs,
        )

    def preprocess_row(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        features = self.featurizer.featurize(
            InputRecord(
                raw_text=row_data.get(RawData.TEXT, ""),
                raw_gazetteer_feats=row_data.get(RawData.DICT_FEAT, ""),
            )
        )
        res = {
            # feature
            ModelInput.WORD_FEAT: self._get_tokens(features),
            ModelInput.DICT_FEAT: (
                features.gazetteer_feats,
                features.gazetteer_feat_weights,
                features.gazetteer_feat_lengths,
            ),
            ModelInput.CHAR_FEAT: features.characters,
            ModelInput.PRETRAINED_MODEL_EMBEDDING: features.pretrained_token_embedding,
            # target
            DocLabelConfig._name: row_data.get(RawData.DOC_LABEL),
            # extra data
            ExtraField.RAW_TEXT: row_data.get(RawData.TEXT),
        }
        if Target.TARGET_PROB_FIELD in self.labels:
            res[Target.TARGET_PROB_FIELD] = parse_json_array(
                row_data[RawData.TARGET_PROBS]
            )
            res[Target.TARGET_LABEL_FIELD] = parse_json_array(
                row_data[RawData.TARGET_LABELS]
            )
            res[Target.TARGET_LOGITS_FIELD] = parse_json_array(
                row_data[RawData.TARGET_LOGITS]
            )

        return res

    def init_target_metadata(
        self,
        train_data: textdata.Dataset,
        eval_data: textdata.Dataset,
        test_data: textdata.Dataset,
    ):
        # build vocabs for label fields
        for name, label in self.labels.items():
            if name in [Target.TARGET_PROB_FIELD, Target.TARGET_LOGITS_FIELD]:
                continue
            # Need test data to make sure we cover all of the labels in it
            # It is particularly important when BIO is enabled as a B-[Label] can
            # appear in train and eval but test can have B-[Label] and I-[Label]

            if label.use_vocab:
                print("Building vocab for label {}".format(name))
                label.build_vocab(train_data, eval_data, test_data)
                print(
                    "{} field's vocabulary size is {}".format(
                        name, len(label.vocab.itos)
                    )
                )

        self.metadata.target = [
            field.get_meta()
            for name, field in self.labels.items()
            if name not in [Target.TARGET_PROB_FIELD, Target.TARGET_LOGITS_FIELD]
        ]
        if len(self.metadata.target) == 1:
            self.metadata.target = self.metadata.target[0]

    def _align_target_label(self, target, label_list, batch_label_list):
        """
        align the target in the order of label_list, batch_label_list stores the
        original target order.
        """
        if sorted(label_list) != sorted(batch_label_list[0]):
            raise Exception(
                "label list %s is not matched with doc label %s",
                (str(batch_label_list), str(label_list)),
            )

        def get_sort_idx(l):
            return [i[0] for i in sorted(enumerate(l), key=lambda x: x[1])]

        def reorder(l, o):
            return [l[i] for i in o]

        unsort_idx = get_sort_idx(get_sort_idx(label_list))
        align_target = [
            reorder(reorder(t, get_sort_idx(b)), unsort_idx)
            for t, b in zip(target, batch_label_list)
        ]
        return align_target

    def _target_from_batch(self, batch):
        label_list = self.metadata.target.vocab.itos
        batch_label_list = getattr(batch, Target.TARGET_LABEL_FIELD)
        targets = []
        for name in self.labels:
            target = getattr(batch, name)
            if name in [Target.TARGET_PROB_FIELD, Target.TARGET_LOGITS_FIELD]:
                target = self._align_target_label(target, label_list, batch_label_list)
            targets.append(target)
        if len(targets) == 1:
            return targets[0]
        return tuple(targets)
