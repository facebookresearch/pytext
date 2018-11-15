#!/usr/bin/env python3
from pytext.config import ConfigBase
from pytext.config.component import create_trainer

from .trainer import Trainer, TrainerBase


class EnsembleTrainer(TrainerBase):
    class Config(ConfigBase):
        real_trainer: Trainer.Config = Trainer.Config()

    @classmethod
    def from_config(cls, config: Config, *args, **kwargs):
        return cls(create_trainer(config.real_trainer, *args, **kwargs))

    def __init__(self, real_trainer):
        self.real_trainer = real_trainer
        self.test = real_trainer.test
        self.train_single_model = real_trainer.train

    def train(self, train_iter, eval_iter, model, *args, **kwargs):
        model.models = [
            self.train_single_model(train_iter, eval_iter, m, *args, **kwargs)
            for m in model.models
        ]
        model.merge_sub_models()
        return model, None
