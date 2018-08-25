#!/usr/bin/env python3
from typing import Union

import torch
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
            # unsqueeze so that we can concat later
            if hasattr(trained_model, "crf") and getattr(trained_model, "crf"):
                model.crf_transition_matrices.append(
                    trained_model.crf.get_transitions().unsqueeze(0)
                )
        # to get the transition_matrix for the ensemble model, we average the
        # transition matrices of the children model
        if hasattr(model, "crf_transition_matrices"):
            transition_matrix = torch.mean(
                torch.cat(model.crf_transition_matrices, dim=0), dim=0
            )
            model.crf.set_transitions(transition_matrix)
        return model
