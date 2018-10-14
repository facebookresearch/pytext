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

    def test(self, model, test_iter, metadata):
        return self.real_trainer.test(model, test_iter, metadata)

    def train(
        self,
        train_iter,
        eval_iter,
        model,
        metrics_reporter,
        optimizers,
        scheduler=None,
    ):
        for i in range(len(model.models)):
            model.models[i], _ = self.train_single_model(
                train_iter,
                eval_iter,
                model.models[i],
                metrics_reporter,
                optimizers,
                scheduler,
            )
        model.merge_sub_models()
        return model

    def train_single_model(
        self,
        train_iter,
        eval_iter,
        model,
        metrics_reporter,
        optimizers,
        scheduler=None,
    ):
        print(f"start training the model")
        return self.real_trainer.train(
            train_iter,
            eval_iter,
            model,
            metrics_reporter,
            optimizers,
            scheduler,
        )
