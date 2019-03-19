#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import time
from typing import Any, Optional, Tuple

import torch
from pytext.common.constants import BatchContext, Stage
from pytext.config import PyTextConfig
from pytext.config.component import (
    Component,
    ComponentType,
    create_optimizer,
    create_scheduler,
)
from pytext.config.pytext_config import ConfigBase
from pytext.data.data_handler import BatchIterator
from pytext.metric_reporters import MetricReporter
from pytext.models.distributed_model import DistributedModel
from pytext.models.model import Model
from pytext.optimizer import Adam, Optimizer, learning_rates
from pytext.optimizer.scheduler import Scheduler
from pytext.utils import cuda, precision, timing


class TrainerBase(Component):
    __COMPONENT_TYPE__ = ComponentType.TRAINER


class TrainingState:
    model: Model
    optimizer: Optimizer
    scheduler: Scheduler
    start_time: float
    epoch: int = 0
    rank: int = 0
    stage: Stage = Stage.TRAIN
    epochs_since_last_improvement: int = 0
    best_model_state: Any = None
    best_model_metric: Any = None

    def __init__(self, **kwargs):
        unknown_keys = kwargs.keys() - TrainingState.__annotations__.keys()
        if unknown_keys:
            raise TypeError(f"TrainingState unexpected attributes {unknown_keys}")
        vars(self).update(kwargs)


class Trainer(TrainerBase):
    """
    Base Trainer class that provide ways to
        1 Train model, compute metrics against eval set and use the metrics for
        model selection.
        2 Test trained model, compute and publish metrics against a blind test set.

    Attributes:
        epochs (int): Training epochs
        early_stop_after (int): Stop after how many epochs when the eval metric
            is not improving
        max_clip_norm (Optional[float]): Clip gradient norm if set
        report_train_metrics (bool): Whether metrics on training data should be
            computed and reported.
        target_time_limit_seconds (float): Target time limit for training in seconds. If
            the expected time to train another epoch exceeds this limit, stop training.
    """

    class Config(ConfigBase):
        #: Training epochs
        epochs: int = 10
        #: Stop after how many epochs when the eval metric is not improving
        early_stop_after: int = 0
        #: Clip gradient norm if set
        max_clip_norm: Optional[float] = None
        #: Whether metrics on training data should be computed and reported.
        report_train_metrics: bool = True
        #: Target time limit for training, default (None) to no time limit.
        target_time_limit_seconds: Optional[int] = None
        #: Whether to do evaluation and model selection based on it.
        do_eval: bool = True
        #: Number of samples for logging training progress.
        num_samples_to_log_progress = 1000
        # config for optimizer, used in parameter update
        optimizer: Optimizer.Config = Adam.Config()
        scheduler: Optional[Scheduler.Config] = None

    def __init__(self, config: Config, model: torch.nn.Module):
        self.optimizer: torch.optim.Optimizer = create_optimizer(
            config.optimizer, model
        )
        self.scheduler: torch.optim.lr_scheduler = (
            create_scheduler(config.scheduler, self.optimizer)
            if config.scheduler
            else Scheduler()
        )

        self.config = config

    @classmethod
    def from_config(cls, config: Config, model: torch.nn.Module, *args, **kwargs):
        return cls(config, model)

    @timing.time("Trainer.test")
    def test(self, test_iter, model, metric_reporter: MetricReporter):
        state = TrainingState(stage=Stage.TEST, model=model, epoch=1)
        if cuda.CUDA_ENABLED:
            state.model.cuda()
        state.model.eval()
        with torch.no_grad():
            return self.run_epoch(state, test_iter, metric_reporter)

    @timing.time("pre-training")
    def set_up_training(self, state: TrainingState, training_data: BatchIterator):
        if cuda.CUDA_ENABLED:
            state.model.cuda()
        state.scheduler.prepare(training_data, self.config.epochs)

        if cuda.DISTRIBUTED_WORLD_SIZE > 1:
            device_id = torch.cuda.current_device()
            state.model = DistributedModel(
                module=state.model,
                device_ids=[device_id],
                output_device=device_id,
                broadcast_buffers=False,
            )

        state.optimizer = precision.wrap_optimizer(state.optimizer)
        state.start_time = time.time()

    @timing.time("zero gradients")
    def zero_grads(self, state):
        if state.stage != Stage.TRAIN:
            return

        if cuda.DISTRIBUTED_WORLD_SIZE > 1:
            # replace optimizer.zero_grad() here to work with DDP
            # in cases where some parameters don't receive grads at each step
            # loss.backward will set grad for params in the computation graph
            # we can thus follow which params are left out and call .backward
            # on them manually
            for p in state.model.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad = None
        else:
            state.optimizer.zero_grad()

    @timing.time("backprop")
    def backprop(self, state, loss):
        if state.stage != Stage.TRAIN:
            return

        with timing.time("loss.backward"):
            precision.backward(state.optimizer, loss)
            if cuda.DISTRIBUTED_WORLD_SIZE > 1:
                # DDP fix when some parameters don't receive grads
                for p in state.model.parameters():
                    if p.requires_grad and p.grad is None:
                        p.backward(torch.zeros_like(p.data))

        state.scheduler.step_batch()

        if self.config.max_clip_norm is not None:
            grad_norm = precision.clip_grad_norm(
                state.model, self.optimizer, self.config.max_clip_norm
            )
        else:
            grad_norm = None

        with timing.time("optimizer.step"):
            state.optimizer.step()
        # grad_norm could be used to check grads sync in distributed training
        return grad_norm

    def continue_training(self, state: TrainingState) -> bool:
        # Are we done?
        if state.epoch >= self.config.epochs:
            return False

        # Check whether the model has improved recently enough
        # Only do this if we're bothering to evaluate the model
        if self.config.do_eval and state.epochs_since_last_improvement >= (
            self.config.early_stop_after or float("inf")
        ):
            print(
                f"Worker {state.rank}: Eval metric hasn't changed for "
                + f"{state.epochs_since_last_improvement} epochs. Stopping now."
            )
            return False

        # Check whether we think the next epoch will put us over the configured
        # time limit.
        epochs_run = state.epoch + 1
        time_elapsed = time.time() - state.start_time
        mean_epoch_time = time_elapsed / epochs_run
        expected_next_epoch_time = time_elapsed + mean_epoch_time
        target_time_limit = (
            float("inf")
            if self.config.target_time_limit_seconds is None
            else self.config.target_time_limit_seconds
        )
        if expected_next_epoch_time > target_time_limit:
            print(
                f"Worker {state.rank}: Stopping training after {epochs_run} epochs "
                f"and {int(time_elapsed)} seconds, due to the target max training "
                f"time of {self.config.target_time_limit_seconds} seconds."
            )
            return False

        return True

    @timing.time("save checkpoint")
    def save_checkpoint(self, state: TrainingState, train_config: PyTextConfig):
        # Only one worker should save checkpoints
        if state.rank != 0:
            return

        print(f"Found a better model!")
        if train_config.save_module_checkpoints:
            state.model.save_modules(
                base_path=train_config.modules_save_dir, suffix=f"-ep{state.epoch}"
            )

        model_state = state.model.state_dict()
        # save to cpu to avoid multiple model copies in gpu memory
        if cuda.CUDA_ENABLED:
            for key, parameter in model_state.items():
                model_state[key] = parameter.cpu()
        state.best_model_state = model_state

    def load_best_model(self, state: TrainingState):
        if cuda.CUDA_ENABLED:
            state.model.load_state_dict(
                {k: v.cuda() for k, v in state.best_model_state.items()}
            )
        else:
            state.model.load_state_dict(state.best_model_state)

    @timing.time("Trainer.train")
    def train(
        self,
        training_data: BatchIterator,
        eval_data: BatchIterator,
        model: Model,
        metric_reporter: MetricReporter,
        train_config: PyTextConfig,
        rank: int = 0,
    ) -> Tuple[torch.nn.Module, Any]:
        """
        Train and eval a model, the model states will be modified. This function
        iterates epochs specified in config, and for each epoch do:

            1. Train model using training data, aggregate and report training results
            2. Adjust learning rate if scheduler is specified
            3. Evaluate model using evaluation data
            4. Calculate metrics based on evaluation results and select best model

        Args:
            train_iter (BatchIterator): batch iterator of training data
            eval_iter (BatchIterator): batch iterator of evaluation data
            model (Model): model to be trained
            metric_reporter (MetricReporter): compute metric based on training
                output and report results to console, file.. etc
            train_config (PyTextConfig): training config
            training_result (Optional): only meaningful for Hogwild training. default
                is None
            rank (int): only used in distributed training, the rank of the current
                training thread, evaluation will only be done in rank 0

        Returns:
            model, best_metric: the trained model together with the best metric
        """
        state = TrainingState(
            model=model, optimizer=self.optimizer, scheduler=self.scheduler, rank=rank
        )
        self.set_up_training(state, training_data)

        while self.continue_training(state):
            state.epoch += 1
            state.epochs_since_last_improvement += 1
            print(f"Worker {state.rank} starting epoch {state.epoch}", flush=True)
            lrs = learning_rates(state.optimizer)
            print(f"Learning rate(s): {', '.join(map(str, lrs))}")

            with timing.time("train epoch"):
                state.stage = Stage.TRAIN
                state.model.train()
                self.run_epoch(state, training_data, metric_reporter)

            if not self.config.do_eval:
                continue

            with timing.time("eval epoch"):
                state.stage = Stage.EVAL
                model.eval(Stage.EVAL)
                with torch.no_grad():
                    eval_metric = self.run_epoch(state, eval_data, metric_reporter)

            # Step the learning rate scheduler(s)
            assert eval_metric is not None
            state.scheduler.step_epoch(
                metrics=metric_reporter.get_model_select_metric(eval_metric),
                epoch=state.epoch,
            )

            # Did we train a better model?
            if metric_reporter.compare_metric(eval_metric, state.best_model_metric):
                state.epochs_since_last_improvement = 0
                state.best_model_metric = eval_metric
                self.save_checkpoint(state, train_config)

        # Only bother loading the best model for master worker
        if rank == 0 and state.best_model_state is not None:
            self.load_best_model(state)

        return state.model, state.best_model_metric

    @timing.report_snapshot
    def run_epoch(
        self, state: TrainingState, data: BatchIterator, metric_reporter: MetricReporter
    ):
        # This method is due for some refactoring, pushing it off because it interacts
        # with the metric reporter too much. Much of the logic here either changes in
        # the NewTaskTrainer or should change with a better metric reporter design.
        report_metric = state.stage != Stage.TRAIN or self.config.report_train_metrics
        model = state.model

        for batch_id, (inputs, targets, context) in enumerate(data):
            self.zero_grads(state)
            # pass context to model to use in forward call if needed
            model.contextualize(context)
            with timing.time("model.forward"):
                logits = model(*inputs)

            with timing.time("compute loss"):
                loss = model.get_loss(logits, targets, context)
                if BatchContext.IGNORE_LOSS in context:
                    loss *= 0

            self.backprop(state, loss)

            if report_metric:
                with timing.time("add metrics"):
                    preds, scores = model.get_pred(
                        logits, targets, context, state.stage, *inputs
                    )
                    metric_reporter.add_batch_stats(
                        batch_id, preds, targets, scores, loss.item(), inputs, **context
                    )

            if (
                state.rank == 0
                and batch_id % self.config.num_samples_to_log_progress == 0
            ):
                print(
                    f"Evaluating batch {batch_id} for epoch {state.epoch}", flush=True
                )

        metrics = None
        if report_metric:
            with timing.time("report metrics"):
                metrics = metric_reporter.report_metric(
                    model, state.stage, state.epoch, print_to_channels=(state.rank == 0)
                )
        else:
            metric_reporter._reset()

        return metrics
