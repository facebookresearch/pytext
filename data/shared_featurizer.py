#!/usr/bin/env python3
import multiprocessing
from typing import Dict, List, Tuple

from assistant.lib.feat.ttypes import ModelFeatures
from caffe2.python.fb.text.raw_feat_config.ttypes import (
    TokenSparseFeatsList,
    TokenSparseFeats,
)
from facebook.assistant.lib.featurization_lib import (
    DEFAULT_LOCALE,
    FeaturizationWrapper,
)
from joblib import Parallel, delayed
from libfb.py.thrift_clients.configerator_thrift_client import ConfigeratorThriftClient
from pytext.common.constants import ConfigeratorPath
from thrift.protocol import TBinaryProtocol, TSimpleJSONProtocol
from thrift.util import Serializer

from .featurizer import Featurizer, InputRecord, TokenFeatures

BIN_FACTORY = TBinaryProtocol.TBinaryProtocolFactory()
JSON_FACTORY = TSimpleJSONProtocol.TSimpleJSONProtocolFactory()
DEFAULT_TOKENIZER_CONFIG = {DEFAULT_LOCALE: ConfigeratorPath.DEFAULT_TOKENIZER}


# raw_dict is expected to be serialized version of
# caffe2.fb.text.TokenSparseFeatsList object
def parse_assistant_raw_record(
    raw_text: str,
    raw_dict: str,
    locale: str = ""
) -> InputRecord:
    if len(raw_dict) > 0:
        token_features = Serializer.deserialize(
            JSON_FACTORY, raw_dict, TokenSparseFeatsList()
        )
    else:
        token_features = TokenSparseFeatsList([])
    return InputRecord(
        raw_text,
        [
            TokenFeatures(sparse_feats.tokenIdx, sparse_feats.features)
            for sparse_feats in token_features.tokenFeatList
        ],
        locale
    )


def build_assistant_raw_dict(token_features: List[TokenFeatures]) -> str:
    sparse_feats_list = TokenSparseFeatsList([
        TokenSparseFeats(token_feature.token_index, token_feature.features)
        for token_feature in token_features
    ])

    return Serializer.serialize(JSON_FACTORY, sparse_feats_list)


class SharedFeaturizer(Featurizer):
    """
    This is a wrapper class over Python binding for Featurizer and, serves as
    an interface for data handler for preprocessing raw input data.
    This interface is specific for Assistant NLU's use-case for now.
    """

    def __init__(
        self,
        tokenizer_config_path_dict: Dict[str, str] = DEFAULT_TOKENIZER_CONFIG,
        sentence_markers_dict: Dict[str, Tuple[str, str]] = None,
        num_threads: int = 0,
        pre_trained_models_dict: Dict[str, str] = None,
    ) -> None:
        tokenizer_config_dict: dict = {}
        with ConfigeratorThriftClient() as ctc:
            for name, config_path in tokenizer_config_path_dict.items():
                tokenizer_config_dict[name] = ctc.getConfigContents(config_path)
        self.add_sentence_markers = False
        if sentence_markers_dict is None:
            if pre_trained_models_dict is None:
                self.featurizer = FeaturizationWrapper(tokenizer_config_dict)
            else:
                self.featurizer = FeaturizationWrapper(
                    tokenizer_config_dict, pre_trained_models_dict
                )
        else:
            self.add_sentence_markers = True
            if pre_trained_models_dict is None:
                self.featurizer = FeaturizationWrapper(
                    tokenizer_config_dict, sentence_markers_dict
                )
            else:
                self.featurizer = FeaturizationWrapper(
                    tokenizer_config_dict,
                    sentence_markers_dict,
                    pre_trained_models_dict,
                )
        self.num_threads = num_threads or multiprocessing.cpu_count()

    def featurize(self, input_record: InputRecord) -> ModelFeatures:
        """Turn Token Features back into format expeced in c++"""
        raw_dict = build_assistant_raw_dict(input_record.token_features)
        """Featurize one instance/example only."""
        features_bin = self.featurizer.featurize_raw_text(
            input_record.raw_text,
            raw_dict,
            input_record.locale,
            self.add_sentence_markers,
        )
        # Get ModelFeatures object
        features_obj = Serializer.deserialize(
            BIN_FACTORY, features_bin, ModelFeatures()
        )
        return features_obj

    def featurize_batch(self, input_records: List[InputRecord]) -> List[ModelFeatures]:
        """Featurize a batch of instances/examples parallelly."""
        raw_feats = [
            (
                input_record.raw_text,
                build_assistant_raw_dict(input_record.token_features),
                input_record.locale,
            )
            for input_record in input_records
        ]
        features_bin_list = self.featurizer.featurize_batch_parallel(
            raw_feats, self.num_threads, self.add_sentence_markers
        )
        # Deserialize ModelFeatures objects
        features_obj_list = Parallel(n_jobs=self.num_threads)(
            delayed(Serializer.deserialize)(BIN_FACTORY, features_bin, ModelFeatures())
            for features_bin in features_bin_list
        )
        return features_obj_list
