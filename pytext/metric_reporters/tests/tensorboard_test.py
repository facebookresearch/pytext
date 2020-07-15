#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from unittest import TestCase

from pytext.common.constants import Stage
from pytext.metric_reporters.channel import TensorBoardChannel
from torch import nn, optim


class FCModelWithNanAndInfWts(nn.Module):
    """ Simple FC model
    """

    def __init__(self):
        super(FCModelWithNanAndInfWts, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc1.weight.data.fill_(float("NaN"))
        self.fc2.weight.data.fill_(float("Inf"))

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)


class TensorboardTest(TestCase):
    def test_report_metrics_with_nan(self):
        """ Check that tensorboard channel catches errors when model has
            Inf or NaN weights
        """
        tensorboard_channel = TensorBoardChannel()
        # create simple model and optimizers
        model = FCModelWithNanAndInfWts()

        optimizer = optim.SGD(model.parameters(), lr=0.1)
        tensorboard_channel.report(
            stage=Stage.TRAIN,
            epoch=1,
            metrics=0.0,
            model_select_metric=0.0,
            loss=1.0,
            preds=[1],
            targets=[1],
            scores=[1],
            context={},
            meta={},
            model=model,
            optimizer=optimizer,
            log_gradient=False,
            gradients={},
        )

    def test_report_metrics_to_others(self):
        """ Check that tensorboard channel catches errors when model has
            Inf or NaN weights
        """
        tensorboard_channel = TensorBoardChannel()
        # create simple model and optimizers
        model = FCModelWithNanAndInfWts()

        optimizer = optim.SGD(model.parameters(), lr=0.1)
        tensorboard_channel.report(
            stage=Stage.OTHERS,
            epoch=1,
            metrics=0.0,
            model_select_metric=0.0,
            loss=1.0,
            preds=[1],
            targets=[1],
            scores=[1],
            context={},
            meta={},
            model=model,
            optimizer=optimizer,
            log_gradient=False,
            gradients={},
        )
