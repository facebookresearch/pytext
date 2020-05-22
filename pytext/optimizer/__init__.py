#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pytext.optimizer.activations import get_activation  # noqa
from pytext.optimizer.fp16_optimizer import (  # noqa
    FP16Optimizer,
    FP16OptimizerApex,
    FP16OptimizerFairseq,
)
from pytext.optimizer.lamb import Lamb  # noqa
from pytext.optimizer.optimizers import (  # noqa
    SGD,
    Adagrad,
    Adam,
    AdamW,
    Optimizer,
    learning_rates,
)
from pytext.optimizer.privacy_engine import PrivacyEngine  # noqa
from pytext.optimizer.radam import RAdam  # noqa
from pytext.optimizer.swa import StochasticWeightAveraging  # noqa
