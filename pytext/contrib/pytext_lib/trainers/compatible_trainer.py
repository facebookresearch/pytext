#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
from contextlib import ExitStack as contextlib_ExitStack
from typing import Any, List, Tuple

import torch
from pytext.common.constants import Stage
from pytext.config import PyTextConfig
from pytext.metric_reporters import ClassificationMetricReporter, MetricReporter
from pytext.metric_reporters.classification_metric_reporter import (
    ComparableClassificationMetric,
)
from pytext.models.model import Model
from pytext.models.roberta import RoBERTa
from pytext.optimizer import Optimizer, learning_rates
from pytext.optimizer.scheduler import Scheduler
from pytext.optimizer.sparsifiers.sparsifier import Sparsifier
from pytext.task.new_task import NewTask
from pytext.task.serialize import save
from pytext.trainers.trainer import TaskTrainer, Trainer, maybe_accumulate_gradients
from pytext.trainers.training_state import TrainingState
from pytext.utils import timing
from torch.utils.data import DataLoader


class CompatibleTrainer(TaskTrainer):
    __EXPANSIBLE__ = True

    class Config(Trainer.Config):
        """Make mypy happy"""

    def __init__(self, model: torch.nn.Module, config: Config = None, **kwargs):
        # temp workaround to minimize changes to TaskTrainer
        if not config:
            config = CompatibleTrainer.Config(**kwargs)

        super().__init__(config, model)

    def train(
        self,
        training_data: DataLoader,
        eval_data: DataLoader,
        model: Model,
        optimizer: Optimizer,
        label_names: List[str],
        scheduler: Scheduler = None,
        sparsifier: Sparsifier = None,
        metric_reporter: MetricReporter = None,
        train_config: PyTextConfig = None,
        rank: int = 0,
    ) -> Tuple[torch.nn.Module, Any]:
        # temp workaround to minimize changes to TaskTrainer
        if not train_config:
            train_config = PyTextConfig(
                task=NewTask.Config(model=RoBERTa.Config), version=20
            )
        if scheduler:
            self.scheduler = scheduler
        if sparsifier:
            self.sparsifier = sparsifier

        state = TrainingState(
            model=model,
            optimizer=optimizer,
            scheduler=self.scheduler,
            sparsifier=self.sparsifier,
            rank=rank,
        )
        metric_reporter_config = ClassificationMetricReporter.Config(
            output_path="/tmp/test_out.txt",
            pep_format=False,
            model_select_metric=ComparableClassificationMetric.ACCURACY,  # in json: "accuracy"
            target_label=None,
            text_column_names=["text"],
            additional_column_names=[],
            recall_at_precision_thresholds=[0.2, 0.4, 0.6, 0.8, 0.9],
        )
        metric_reporter = ClassificationMetricReporter.from_config_and_label_names(
            config=metric_reporter_config, label_names=label_names
        )
        return self.train_from_state(
            state, training_data, eval_data, metric_reporter, train_config
        )

    @timing.time("Trainer.train_from_state")
    def train_from_state(
        self,
        state: TrainingState,
        training_data: DataLoader,
        eval_data: DataLoader,
        metric_reporter: MetricReporter,
        train_config: PyTextConfig,
    ) -> Tuple[torch.nn.Module, Any]:
        """
        Train and eval a model from a given training state will be modified.
        This function iterates epochs specified in config, and for each epoch do:

            1. Train model using training data, aggregate and report training results
            2. Adjust learning rate if scheduler is specified
            3. Evaluate model using evaluation data
            4. Calculate metrics based on evaluation results and select best model

        Args:
            training_state (TrainingState): contrains stateful information to be
            able to restore a training job
            train_iter (DataLoader): batch iterator of training data
            eval_iter (DataLoader): batch iterator of evaluation data
            model (Model): model to be trained
            metric_reporter (MetricReporter): compute metric based on training
                output and report results to console, file.. etc
            train_config (PyTextConfig): training config

        Returns:
            model, best_metric: the trained model together with the best metric
        """
        training_data = self.set_up_training(state, training_data)
        model = state.model
        rank = state.rank
        trainable_params = sum(
            p.numel() for p in state.model.parameters() if p.requires_grad
        )
        print(f"Model :{model}")
        print(f"Num trainable parameters: {trainable_params}")

        while self.continue_training(state):
            state.epoch += 1
            state.epochs_since_last_improvement += 1
            lrs = learning_rates(state.optimizer)
            print(f"\nWorker {state.rank} starting epoch {state.epoch}")
            print(f"Learning rate(s): {', '.join(map(str, lrs))}")

            with timing.time("train epoch"):
                state.stage = Stage.TRAIN
                state.model.train()
                print(f"start training epoch {state.epoch}")
                epoch_data = training_data
                if self.config.num_batches_per_epoch:
                    # We want to limit the number of batches in the epoch;
                    # equivalent to epoch_data[:num_batches_per_epoch] for iterators.
                    # In this case we set the training data iterator to cycle earlier
                    # in the training process, so when it reaches the end it will
                    # loop back to the beginning.
                    epoch_data = itertools.islice(
                        epoch_data, self.config.num_batches_per_epoch
                    )
                self.run_epoch(state, epoch_data, metric_reporter)

            if not self.config.do_eval:
                continue

            with timing.time("eval epoch"):
                state.stage = Stage.EVAL
                model.eval()
                print(f"start evaluating epoch {state.epoch}")
                with torch.no_grad():
                    eval_metric = self.run_epoch(state, eval_data, metric_reporter)

            # Step the learning rate scheduler(s)
            assert eval_metric is not None
            state.scheduler.step_epoch(
                metrics=metric_reporter.get_model_select_metric(eval_metric),
                epoch=state.epoch,
            )

            # Did we train a better model?
            better_model = metric_reporter.compare_metric(
                eval_metric, state.best_model_metric
            )
            if better_model:
                self.update_best_model(state, train_config, eval_metric)
            if better_model or train_config.save_all_checkpoints:
                self.save_checkpoint(state, train_config)

        if self.optimizer.finalize():
            should_update_model = True
            eval_metric = None
            if self.config.do_eval:
                state.stage = Stage.EVAL
                model.eval()
                print("start evaluating finalized state")
                with torch.no_grad():
                    eval_metric = self.run_epoch(state, eval_data, metric_reporter)
                should_update_model = metric_reporter.compare_metric(
                    eval_metric, state.best_model_metric
                )
            if should_update_model:
                self.update_best_model(state, train_config, eval_metric)
            if should_update_model or train_config.save_all_checkpoints:
                self.save_checkpoint(state, train_config)
        # Only bother loading the best model for master worker
        if (
            rank == 0
            and state.best_model_state is not None
            and self.config.load_best_model_after_train
        ):
            self.load_best_model(state)

        return state.model, state.best_model_metric

    @timing.time("run_step")
    def run_step(
        self,
        samples: List[Any],
        state: TrainingState,
        metric_reporter: MetricReporter,
        report_metric: bool,
    ):
        """Our run_step is a bit different, because we're wrapping the model forward
        call with model.train_batch, which arranges tensors and gets loss, etc.

        Whenever "samples" contains more than one mini-batch (sample_size > 1),
        we want to accumulate gradients locally and only call all-reduce in the
        last backwards pass.
        """
        sample_size = len(samples)
        assert sample_size <= self.config.num_accumulated_batches

        model = state.model
        self.zero_grads(state)
        for idx, (batch_id, batch) in enumerate(samples):
            with contextlib_ExitStack() as exit_stack:
                # enter ddp no_sync context and fp16 delay_scale context if needed
                maybe_accumulate_gradients(exit_stack, model, idx, sample_size)
                logits = model(batch)
                targets = batch["label_ids"]
                loss = model.get_loss(logits, targets)
                if sample_size > 1:
                    # gradients averaged per batch and accumulated across samples.
                    # divide sample_size to let gradients averaged per example
                    loss = loss / sample_size
                self.backprop(state, loss)

            if report_metric:
                with timing.time("add metrics"):
                    predictions, scores = model.get_pred(logits)
                    # [len(targets)] means the batch_size, it's required by add_batch_stats
                    # Will rewrite metric_reporter rather than fixing it
                    metric_data = (predictions, targets, scores, loss, [targets])
                    metric_reporter.add_batch_stats(
                        batch_id,
                        *metric_data,
                        # TODO merge this step into add_batch_stats once all data
                        # migration is done
                        # in new data API, we don't have raw_batch
                        **metric_reporter.batch_context(raw_batch=[], batch=batch),
                    )
                if batch_id % self.config.num_samples_to_log_progress == 0:
                    metric_reporter.report_realtime_metric(state.stage)
        # update gradients after #len(samples) forward & backward
        self.optimizer_step(state)
        self.sparsification_step(state)

    @timing.time("save checkpoint")
    def save_checkpoint(self, state: TrainingState, train_config: PyTextConfig) -> str:
        # Only one worker should save checkpoints
        if state.rank != 0:
            return
        # checkpoint the whole model instead of sub-modules
        # users can load sub-modules from model, externally
        if train_config.save_all_checkpoints:
            return save(
                config=train_config,
                model=state.model,
                meta=None,
                tensorizers=None,
                training_state=state,
                identifier=str(state.epoch),
            )
