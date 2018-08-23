#!/usr/bin/env python3
from typing import Union

from pytext.config.component import create_trainer
from pytext.config.pytext_config import ConfigBase

from .classifier_trainer import ClassifierTrainer
from .joint_trainer import JointTrainer
from .tagger_trainer import TaggerTrainer
from .trainer import Trainer


class EnsembleTrainer(Trainer):
    class Config(ConfigBase):
        real_trainer: Union[
            ClassifierTrainer.Config, TaggerTrainer.Config, JointTrainer.Config
        ]

    def __init__(self, config, **kwargs):
        self.real_trainer = create_trainer(config.real_trainer, **kwargs)

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
