#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict

from pytext.config import ConfigBase
from pytext.config.field_config import FeatureConfig, FloatVectorConfig
from pytext.exporters.exporter import ModelExporter
from pytext.fields import FieldMeta


class DenseFeatureExporter(ModelExporter):
    """
    Exporter for models that have DenseFeatures as input to the decoder
    """

    @classmethod
    def get_feature_metadata(
        cls, feature_config: FeatureConfig, feature_meta: Dict[str, FieldMeta]
    ):
        # add all features EXCEPT dense features. The features exported here
        # go through the representation layer
        (
            input_names_rep,
            dummy_model_input_rep,
            feature_itos_map_rep,
        ) = cls._get_exportable_metadata(
            lambda x: isinstance(x, ConfigBase)
            and not isinstance(x, FloatVectorConfig),
            feature_config,
            feature_meta,
        )

        # need feature lengths only for non-dense features
        cls._add_feature_lengths(input_names_rep, dummy_model_input_rep)

        # add dense features. These features don't go through the representation
        # layer, instead they go directly to the decoder
        (
            input_names_dense,
            dummy_model_input_dense,
            feature_itos_map_dense,
        ) = cls._get_exportable_metadata(
            lambda x: isinstance(x, FloatVectorConfig), feature_config, feature_meta
        )

        feature_itos_map_rep.update(feature_itos_map_dense)
        return (
            input_names_rep + input_names_dense,
            tuple(dummy_model_input_rep + dummy_model_input_dense),
            feature_itos_map_rep,
        )
