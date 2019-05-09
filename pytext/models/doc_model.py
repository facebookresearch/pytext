#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Union

import torch
from pytext.config import ConfigBase
from pytext.data.tensorizers import NumericLabelTensorizer
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.model import Model, SimpleTokenModel
from pytext.models.output_layers import ClassificationOutputLayer, RegressionOutputLayer
from pytext.models.output_layers.doc_classification_output_layer import (
    BinaryClassificationOutputLayer,
    MulticlassOutputLayer,
)
from pytext.models.representations.bilstm_doc_attention import BiLSTMDocAttention
from pytext.models.representations.docnn import DocNNRepresentation
from pytext.models.representations.pure_doc_attention import PureDocAttention


class DocModel_Deprecated(Model):
    """
    An n-ary document classification model. It can be used for all text
    classification scenarios. It supports :class:`~PureDocAttention`,
    :class:`~BiLSTMDocAttention` and :class:`~DocNNRepresentation` as the ways
    to represent the document followed by multi-layer perceptron (:class:`~MLPDecoder`)
    for projecting the document representation into label/target space.

    It can be instantiated just like any other :class:`~Model`.

    DEPRECATED: Use NewDocModel instead
    """

    class Config(ConfigBase):
        representation: Union[
            PureDocAttention.Config,
            BiLSTMDocAttention.Config,
            DocNNRepresentation.Config,
        ] = BiLSTMDocAttention.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )


class NewDocModel(SimpleTokenModel):
    """
    An n-ary document classification model. It can be used for all text
    classification scenarios. It supports :class:`~PureDocAttention`,
    :class:`~BiLSTMDocAttention` and :class:`~DocNNRepresentation` as the ways
    to represent the document followed by multi-layer perceptron (:class:`~MLPDecoder`)
    for projecting the document representation into label/target space.

    It can be instantiated just like any other :class:`~Model`.
    """

    class Config(SimpleTokenModel.Config):
        representation: Union[
            PureDocAttention.Config,
            BiLSTMDocAttention.Config,
            DocNNRepresentation.Config,
        ] = BiLSTMDocAttention.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )


class NewDocRegressionModel(NewDocModel):
    """
    Model that's compatible with the new Model abstraction, and is configured for
    regression tasks (specifically for labels, predictions, and loss).
    """

    class Config(NewDocModel.Config):
        class RegressionModelInput(SimpleTokenModel.Config.ModelInput):
            labels: NumericLabelTensorizer.Config = NumericLabelTensorizer.Config()

        inputs: RegressionModelInput = RegressionModelInput()
        output_layer: RegressionOutputLayer.Config = RegressionOutputLayer.Config()
