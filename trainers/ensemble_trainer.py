#!/usr/bin/env python3
from typing import Union

from pytext.config import ConfigBase
from pytext.config.component import create_trainer

from .classifier_trainer import ClassifierTrainer
from .joint_trainer import JointTrainer
from .tagger_trainer import TaggerTrainer
from .trainer import Trainer


class EnsembleTrainer(Trainer):
    class Config(ConfigBase):
        real_trainer: Union[
            ClassifierTrainer.Config, TaggerTrainer.Config, JointTrainer.Config
        ]

    @classmethod
    def from_config(cls, config: Config, *args, **kwargs):
        return cls(create_trainer(config.real_trainer, *args, **kwargs))

    def __init__(self, real_trainer):
        self.real_trainer = real_trainer

    def test(self, model, test_iter, metadata):
        return self.real_trainer.test(model, test_iter, metadata)

    def train(
        self,
        train_iter,
        eval_iter,
        model,
        optimizers,
        class_names,
        metrics_reporter=None,
        scheduler=None,
    ):
        for i in range(len(model.models)):
            model.models[i] = self.train_single_model(
                train_iter,
                eval_iter,
                model.models[i],
                optimizers,
                class_names,
                metrics_reporter,
                scheduler,
            )
        model.merge_sub_models()
        return model

    def train_single_model(
        self,
        train_iter,
        eval_iter,
        model,
        optimizers,
        class_names,
        metrics_reporter=None,
        scheduler=None,
    ):
        print(f"start training the model")
        trained_model = self.real_trainer.train(
            train_iter,
            eval_iter,
            model,
            optimizers,
            class_names,
            metrics_reporter,
            scheduler,
        )
        return trained_model
