#!/usr/bin/env python3
from typing import Dict, List, Tuple

from assistant.lib.feat.ttypes import ModelFeatures
from facebook.assistant.lib.featurization_lib import (
    DEFAULT_LOCALE,
    FeaturizationWrapper,
)
from joblib import Parallel, delayed
from libfb.py.thrift_clients.configerator_thrift_client import ConfigeratorThriftClient
from pytext.common.constants import ConfigeratorPath
from thrift.protocol import TBinaryProtocol
from thrift.util import Serializer


BIN_FACTORY = TBinaryProtocol.TBinaryProtocolFactory()
DEFAULT_TOKENIZER_CONFIG = {DEFAULT_LOCALE: ConfigeratorPath.DEFAULT_TOKENIZER}


class SharedFeaturizer(object):
    def __init__(
        self, tokenizer_config_path_dict: Dict = DEFAULT_TOKENIZER_CONFIG
    ) -> None:
        tokenizer_configs: dict = {}
        with ConfigeratorThriftClient() as ctc:
            for name, config_path in tokenizer_config_path_dict.items():
                tokenizer_configs[name] = ctc.getConfigContents(config_path)
        self.featurizer = FeaturizationWrapper(tokenizer_configs)

    # raw_dict is expected to be serialized version of
    # caffe2.fb.text.TokenSparseFeatsList object
    def featurize(
        self, raw_text: str, raw_dict: str, locale: str = ""
    ) -> ModelFeatures:
        features_bin = self.featurizer.featurize_raw_text(raw_text, raw_dict, locale)
        # Get ModelFeatures object
        features_obj = Serializer.deserialize(
            BIN_FACTORY, features_bin, ModelFeatures()
        )

        return features_obj

    def featurize_parallel(
        self, raw_feats: List[Tuple[str, ...]], num_threads: int
    ) -> List[ModelFeatures]:
        if len(raw_feats) > 0 and len(raw_feats[0]) == 2:
            # Set locale to empty string so that default locale is used.
            raw_feats = list(map(lambda tup: (tup[0], tup[1], ""), raw_feats))
        features_bin_list = self.featurizer.featurize_batch_parallel(
            raw_feats, num_threads
        )
        # Deserialize ModelFeatures objects
        features_obj_list = Parallel(n_jobs=num_threads)(
            delayed(Serializer.deserialize)(BIN_FACTORY, features_bin, ModelFeatures())
            for features_bin in features_bin_list
        )
        return features_obj_list
