#!/usr/bin/env python3
from typing import Union

from pytext.common.registry import TRAINER, component, create_trainer
from pytext.config.pytext_config import ConfigBase

from .classifier_trainer import ClassifierTrainerConfig
from .joint_trainer import JointTrainerConfig
from .tagger_trainer import TaggerTrainerConfig
from .trainer import Trainer, TrainerConfig


# TODO not the most generic way, need to revisit this part, maybe add TypeVar support
# in ConfigBase
class EnsembleTrainerConfig(ConfigBase, TrainerConfig):
    real_trainer: Union[
        ClassifierTrainerConfig, TaggerTrainerConfig, JointTrainerConfig
    ]


@component(TRAINER, config_cls=EnsembleTrainerConfig)
class EnsembleTrainer(Trainer):
    @classmethod
    def from_config(cls, config: EnsembleTrainerConfig, **metadata):
        return cls(create_trainer(config.real_trainer, **metadata))

    def __init__(self, trainer):
        self.real_trainer = trainer

    def test(self, model, test_iter, metadata):
        return self.real_trainer.test(model, test_iter, metadata)

    def train(
        self,
        train_iter,
        eval_iter,
        model,
        optimizers,
        loss_fn,
        class_names,
        metrics_reporter=None,
    ):
        for i in range(len(model.models)):
            print(f"start training the {i} model")
            trained_model = self.real_trainer.train(
                train_iter,
                eval_iter,
                model.models[i],
                optimizers,
                loss_fn,
                class_names,
                metrics_reporter,
            )
            model.models[i] = trained_model
        return model
