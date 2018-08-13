#!/usr/bin/env python3
from libfb.py.thrift_clients.configerator_thrift_client import ConfigeratorThriftClient
from thrift.util import Serializer
from assistant.lib.feat.ttypes import ModelFeatures
from facebook.assistant.lib.featurization_lib import FeaturizationWrapper
from thrift.protocol import TSimpleJSONProtocol, TBinaryProtocol
from pytext.common.constants import ConfigeratorPath
from typing import List, Tuple

json_factory = TSimpleJSONProtocol.TSimpleJSONProtocolFactory()
bin_factory = TBinaryProtocol.TBinaryProtocolFactory()
from joblib import Parallel, delayed


class SharedFeaturizer(object):
    def __init__(self, config_path: str = ConfigeratorPath.DEFAULT_TOKENIZER) -> None:
        with ConfigeratorThriftClient() as ctc:
            tokenizer_config = ctc.getConfigContents(config_path)
            self.featurizer = FeaturizationWrapper(tokenizer_config)

    # raw_dict is expected to be serialized version of
    # caffe2.fb.text.TokenSparseFeatsList object
    def featurize(self, raw_text: str, raw_dict: str) -> ModelFeatures:
        features_bin = self.featurizer.featurize_raw_text(raw_text, raw_dict)
        # Get ModelFeatures object
        features_obj = Serializer.deserialize(
            bin_factory, features_bin, ModelFeatures()
        )

        return features_obj

    def featurize_parallel(
        self, raw_feats: List[Tuple[str, str]], num_threads: int
    ) -> List[ModelFeatures]:
        features_bin_list = self.featurizer.featurize_batch_parallel(
            raw_feats, num_threads
        )
        # Deserialize ModelFeatures objects
        features_obj_list = Parallel(n_jobs=num_threads)(
            delayed(Serializer.deserialize)(bin_factory, features_bin, ModelFeatures())
            for features_bin in features_bin_list
        )
        return features_obj_list
