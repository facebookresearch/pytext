#!/usr/bin/env python3

from typing import Dict, List, Type

import pandas as pd
import torch
from facebook.assistant.lib.featurization_lib import DEFAULT_LOCALE
from fairseq.data.dictionary import Dictionary as FairseqDict
from pytext.common.constants import DatasetFieldName, DFColumn, VocabMeta
from pytext.config import ConfigBase
from pytext.config.component import create_featurizer
from pytext.config.field_config import FeatureConfig, LabelConfig
from pytext.data.featurizer import Featurizer, InputRecord
from pytext.fb.data.assistant_featurizer import AssistantFeaturizer
from pytext.fields import Field, TextFeatureField
from pytext.utils.cuda_utils import GetTensor

from .data_handler import CommonMetadata, DataHandler


WHITE_SPACE_TOKENIZER_CONFIG = '{"newRegexTokenizerConfig": {}}'


class FairSeqMetadata(CommonMetadata):
    source_dict: FairseqDict
    target_dict: FairseqDict


class FairSeqDataHandler(DataHandler):
    class Config(ConfigBase, DataHandler.Config):
        columns_to_read: List[str] = [
            DFColumn.SOURCE_SEQUENCE,
            DFColumn.TARGET_SEQUENCE,
        ]
        target_featurizer: AssistantFeaturizer.Config = AssistantFeaturizer.Config(
            tokenizer_config_dict={DEFAULT_LOCALE: WHITE_SPACE_TOKENIZER_CONFIG}
        )

    def __init__(
        self, featurizer: Featurizer, target_featurizer: Featurizer, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.featurizer = featurizer
        self.target_featurizer = target_featurizer
        # Defines how to map columns in the data-frame after preprocessing to
        # the dataset fields so they can be accessed in the BatchIterator as
        # batch.field_name
        self.df_to_example_func_map = {
            # Source sequence
            DatasetFieldName.SOURCE_SEQ_FIELD: lambda row, field: row[
                DFColumn.SOURCE_FEATS
            ].tokens,
            # Target sequence
            DatasetFieldName.TARGET_SEQ_FIELD: lambda row, field: row[
                DFColumn.TARGET_TOKENS
            ].tokens,
        }
        self.metadata_cls: Type = FairSeqMetadata
        self.metadata: FairSeqMetadata = FairSeqMetadata()

    @classmethod
    def from_config(
        cls,
        config: Config,
        feature_config: FeatureConfig,
        label_config: LabelConfig,
        **kwargs
    ):
        columns = config.columns_to_read
        # Source Sequence features
        features: Dict[str, Field] = {
            DatasetFieldName.SOURCE_SEQ_FIELD: TextFeatureField(
                eos_token=VocabMeta.EOS_TOKEN
            )
        }
        # ToDo: rename labels to targets
        labels: Dict[str, Field] = {
            DatasetFieldName.TARGET_SEQ_FIELD: TextFeatureField(
                eos_token=VocabMeta.EOS_TOKEN
            )
        }
        return cls(
            featurizer=create_featurizer(config.featurizer, feature_config),
            target_featurizer=create_featurizer(
                config.target_featurizer, feature_config
            ),
            raw_columns=columns,
            features=features,
            labels=labels,
            text_feature_name=DatasetFieldName.SOURCE_SEQ_FIELD,
        )

    def _gen_extra_metadata(self) -> None:
        self.metadata.source_dict = self._field_to_fairseq_dict(
            self.features[DatasetFieldName.SOURCE_SEQ_FIELD]
        )
        self.metadata.target_dict = self._field_to_fairseq_dict(
            self.labels[DatasetFieldName.TARGET_SEQ_FIELD]
        )

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Featurize source sequence
        df[DFColumn.RAW_FEATS] = df.apply(
            lambda row: InputRecord(raw_text=row[DFColumn.SOURCE_SEQUENCE]), axis=1
        )

        df[DFColumn.SOURCE_FEATS] = pd.Series(
            self.featurizer.tokenize_batch(df[DFColumn.RAW_FEATS].tolist())
        )

        if DFColumn.TARGET_SEQUENCE in df:
            df[DFColumn.RAW_FEATS] = df.apply(
                lambda row: InputRecord(raw_text=row[DFColumn.TARGET_SEQUENCE]), axis=1
            )
            df[DFColumn.TARGET_TOKENS] = pd.Series(
                self.target_featurizer.tokenize_batch(df[DFColumn.RAW_FEATS].tolist())
            )
        return df

    def _train_input_from_batch(self, batch):
        # source_seqs, source_lens and shifted version of the targets
        # to be fed to the decoder as the prev output tokens
        return (
            batch.source_sequence[0],
            batch.source_sequence[1],
            self._shift_target(
                batch.target_sequence[0],
                self.labels[DatasetFieldName.TARGET_SEQ_FIELD].get_meta().eos_token_idx,
            ),
        )

    def _test_input_from_batch(self, batch):
        # source_seqs, source_lens and None for the shifted version of the targets
        return (batch.source_sequence[0], batch.source_sequence[1], None)

    def _target_from_batch(self, batch):
        return (batch.target_sequence[0], batch.target_sequence[1])

    def _shift_target(self, in_sequences, eos_idx):
        shifted_sequence = GetTensor(torch.LongTensor(in_sequences.size()))
        for i, in_seq in enumerate(in_sequences):
            shifted_sequence[i, :-1] = eos_idx
            shifted_sequence[i, 1:] = in_seq[:-1]
        return shifted_sequence

    @staticmethod
    def _field_to_fairseq_dict(field):
        fs_dict = FairseqDict()
        fs_dict.unk_word = (VocabMeta.UNK_TOKEN,)
        fs_dict.pad_word = VocabMeta.PAD_TOKEN
        fs_dict.eos_word = VocabMeta.EOS_TOKEN
        fs_dict.symbols = field.vocab.itos
        fs_dict.count = [field.vocab.freqs[s] for s in fs_dict.symbols]
        fs_dict.indices = field.vocab.stoi
        return fs_dict
