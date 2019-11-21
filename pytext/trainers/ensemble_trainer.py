#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from pytext.config import ConfigBase
from pytext.config.component import create_trainer

from .trainer import TaskTrainer, TrainerBase


class EnsembleTrainer(TrainerBase):
    """Trainer for ensemble models

    Attributes:
        real_trainer (Trainer): the actual trainer to run
    """

    class Config(ConfigBase):
        real_trainer: TaskTrainer.Config = TaskTrainer.Config()

    @classmethod
    def from_config(cls, config: Config, model: torch.nn.Module, *args, **kwargs):
        trainers = [
            create_trainer(config.real_trainer, m, *args, **kwargs)
            for m in model.models
        ]

        return cls(trainers)

    def __init__(self, real_trainers):
        self.real_trainers = real_trainers
        self.optimizer = real_trainers[0].optimizer
        self.test = real_trainers[0].test
        self.train_single_model = real_trainers[0].train

    """
    Train and eval ensemble model, each sub model will be trained separately in
    sequence, and ``model.merge_sub_models`` will be called to merge states in sub
    models (e.g transition matrix for crf). To train sub models in parallel, please
    use train_single_model method instead

    Args:
        train_iter (BatchIterator): batch iterator of training data
        eval_iter (BatchIterator): batch iterator of evaluation data
        model (Model): model to be trained
        metric_reporter (MetricReporter): compute metric based on training
            output and report results to console, file.. etc
        train_config (PyTextConfig): training config
        optimizers (List[torch.optim.Optimizer]): a list of torch optimizers, in
            most of the case only contains one optimizer
        scheduler (Optional[torch.optim.lr_scheduler]): learning rate scheduler,
            default is None
        rank (int): only used in distributed training, the rank of the current
            training thread, evaluation will only be done in rank 0

    Returns:
        model, none: only the trained ensemble model, no best metric will be returned
            since there's no clear way of aggregating metric from sub models
    """

    def train(self, train_iter, eval_iter, model, *args, **kwargs):
        for (m, t) in zip(model.models, self.real_trainers):
            t.train(train_iter, eval_iter, m, *args, **kwargs)
        model.merge_sub_models()
        return model, None
