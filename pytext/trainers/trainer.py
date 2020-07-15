#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import time
from contextlib import ExitStack as contextlib_ExitStack
from typing import Any, Iterable, List, Optional, Tuple

import torch
from pytext.common.constants import BatchContext, Stage
from pytext.config import PyTextConfig
from pytext.config.component import (
    Component,
    ComponentType,
    create_optimizer,
    create_privacy_engine,
    create_scheduler,
    create_sparsifier,
)
from pytext.config.pytext_config import ConfigBase
from pytext.data.data_handler import BatchIterator
from pytext.metric_reporters import MetricReporter
from pytext.models.distributed_model import DistributedModel
from pytext.models.model import Model
from pytext.optimizer import Adam, Optimizer, PrivacyEngine, learning_rates
from pytext.optimizer.fp16_optimizer import FP16Optimizer, FP16OptimizerFairseq
from pytext.optimizer.scheduler import Scheduler
from pytext.optimizer.sparsifiers.sparsifier import Sparsifier
from pytext.task.serialize import save
from pytext.trainers.training_state import TrainingState
from pytext.utils import cuda, distributed, precision, timing


class TrainerBase(Component):
    __COMPONENT_TYPE__ = ComponentType.TRAINER


def cycle(iterator: Iterable[Any]) -> Iterable[Any]:
    """Like itertools.cycle, but will call iter on the original iterable instead.
    This limits it to not be able to run on say raw generators, but also doesn't
    store a copy of the iterable in memory for repetition."""
    while True:
        yield from iterator


def maybe_accumulate_gradients(exit_stack, model, index, sample_size):
    # index == sample_size - 1 represents the last backward pass
    if (
        cuda.DISTRIBUTED_WORLD_SIZE > 1
        and hasattr(model, "no_sync")
        and index < sample_size - 1
    ):
        """
        Whenever *samples* contains more than one mini-batch (e.g sample_size > 1),
        we want to accumulate gradients locally and only call all-reduce in the
        last backwards pass.
        """
        exit_stack.enter_context(model.no_sync())

    if precision.FP16_ENABLED and index < sample_size - 1:
        """
        Whenever *samples* contains more than one mini-batch (e.g sample_size > 1),
        we want to accumulate gradients in FP16 parameters (e.g delay unscale)
        and only unscale to FP32 parameters after the last backward pass.
        """
        exit_stack.enter_context(precision.delay_unscale())


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
        #: if do_eval, do we load the best model state dict after training or just
        # use the latest model state
        load_best_model_after_train: bool = True
        #: Number of samples for logging training progress.
        num_samples_to_log_progress: int = 1000
        #: Number of forward & backward per batch before update gradients, the
        #: actual_batch_size = batch_size x num_accumulated_batches
        num_accumulated_batches: int = 1
        #: Define epoch as a fixed number of batches. Subsequent epochs will continue
        #: to iterate through the data, cycling through it when they reach the end.
        #: If not set, use exactly one pass through the dataset as one epoch.
        #: This configuration only affects the train epochs, test and eval
        #: will always test their entire datasets.
        num_batches_per_epoch: Optional[int] = None
        #: config for optimizer, used in parameter update
        optimizer: Optimizer.Config = Adam.Config()
        scheduler: Optional[Scheduler.Config] = None
        sparsifier: Optional[Sparsifier.Config] = None
        #: Define arguments for fp16 training. A fp16_optimizer will be created
        #: and wraps the original optimizer, which will scale loss during
        #: backward and master weight will be maintained on original optimizer.
        #: https://arxiv.org/abs/1710.03740
        fp16_args: FP16Optimizer.Config = FP16OptimizerFairseq.Config()
        # PrivacyEngine related args
        privacy_engine: Optional[PrivacyEngine.Config] = None

    def __init__(self, config: Config, model: torch.nn.Module):
        if config.early_stop_after > 0:
            assert config.do_eval, "can't do early stopping when not running evalution"

        if precision.FP16_ENABLED:
            self.optimizer: torch.optim.Optimizer = create_optimizer(
                config.fp16_args,
                model,
                config.optimizer,
                config.num_accumulated_batches,
            )
        else:
            self.optimizer: torch.optim.Optimizer = create_optimizer(
                config.optimizer, model
            )
        self.privacy_engine: PrivacyEngine = (
            create_privacy_engine(config.privacy_engine, model, self.optimizer)
            if config.privacy_engine
            else None
        )

        self.scheduler: torch.optim.lr_scheduler = (
            create_scheduler(config.scheduler, self.optimizer)
            if config.scheduler
            else Scheduler()
        )
        self.sparsifier: Sparsifier = (
            create_sparsifier(config.sparsifier) if config.sparsifier else Sparsifier()
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
                find_unused_parameters=state.model.find_unused_parameters,
                process_group=distributed._round_robin_process_group,
            )
        state.start_time = time.time()

        if self.config.num_batches_per_epoch:
            # Set the training_data iterator to cycle, so it will never run out,
            # but rather after reaching the end will loop back to the beginning.
            training_data = cycle(training_data)
        return training_data

    @timing.time("zero gradients")
    def zero_grads(self, state):
        if state.stage != Stage.TRAIN:
            return
        state.optimizer.zero_grad()

    @timing.time("backprop")
    def backprop(self, state, loss):
        if state.stage != Stage.TRAIN:
            return

        with timing.time("loss.backward"):
            state.optimizer.backward(loss)

    @timing.time("optimizer")
    def optimizer_step(self, state):
        if state.stage != Stage.TRAIN:
            return

        try:
            grad_norm = state.optimizer.clip_grad_norm(
                self.config.max_clip_norm, state.model
            )
        except OverflowError as e:
            print(f"Gradient overflow. Skipping step, {e}")
            return None

        state.scheduler.step_batch()
        with timing.time("optimizer.step"):
            state.optimizer.step()

        state.step_counter += 1
        # grad_norm could be used to check grads sync in distributed training
        return grad_norm

    @timing.time("sparsifier")
    def sparsification_step(self, state):
        # sparsification only if sparifier is used
        if not self.config.sparsifier:
            return

        self.sparsifier.sparsify(state)
        if state.rank == 0:
            current_sparsity = self.sparsifier.get_current_sparsity(state.model)
            print(f"sparsity in the model: {current_sparsity}")

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

    def move_state_dict_to_cpu(self, state_dict):
        for key, maybe_parameter in state_dict.items():
            if isinstance(maybe_parameter, torch.Tensor):
                state_dict[key] = maybe_parameter.cpu()
            else:
                self.move_state_dict_to_cpu(maybe_parameter)
        return state_dict

    def move_state_dict_to_gpu(self, state_dict):
        for key, maybe_parameter in state_dict.items():
            if isinstance(maybe_parameter, torch.Tensor):
                state_dict[key] = maybe_parameter.cuda()
            else:
                self.move_state_dict_to_gpu(maybe_parameter)
        return state_dict

    def update_best_model(
        self, state: TrainingState, train_config: PyTextConfig, eval_metric
    ):
        # This should be updated by all workers so they agree on when to stop training
        # when `early_stop_after` is specified.
        state.epochs_since_last_improvement = 0
        state.best_model_metric = eval_metric
        print(f"Found a better model!")

        # Only one worker should save checkpoints
        if state.rank != 0:
            return

        model_state = state.model.state_dict()
        # save to cpu to avoid multiple model copies in gpu memory
        if cuda.CUDA_ENABLED:
            self.move_state_dict_to_cpu(model_state)
        state.best_model_state = model_state

    @timing.time("save checkpoint")
    def save_checkpoint(self, state: TrainingState, train_config: PyTextConfig) -> str:
        # Only one worker should save checkpoints
        if state.rank != 0:
            return

        if train_config.save_module_checkpoints or train_config.save_all_checkpoints:
            # saves per-epoch sub-modules when save_all_checkpoints or
            # save_module_checkpoints is enabled
            state.model.save_modules(
                base_path=train_config.modules_save_dir, suffix=f"-ep{state.epoch}"
            )
        if state.epochs_since_last_improvement == 0:
            # state.epochs_since_last_improvement == 0 means found a better
            # model in current epoch, thus update best model's sub-modules
            state.model.save_modules(base_path=train_config.modules_save_dir)

        # next to add new config and implementation of frequency on checkpointing
        if train_config.save_all_checkpoints:
            return save(
                config=train_config,
                model=state.model,
                meta=None,
                tensorizers=None,
                training_state=state,
                identifier=str(state.epoch),
            )

    def load_best_model(self, state: TrainingState):
        if cuda.CUDA_ENABLED:
            # Move current model to CPU to avoid multiple models in GPU memory
            state.model.cpu()
            state.model.load_state_dict(state.best_model_state)
            # Move model back to GPU
            state.model.cuda()
        else:
            state.model.load_state_dict(state.best_model_state)

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
        Train and eval a model, the model states will be modified.
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
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sparsifier=self.sparsifier,
            privacy_engine=self.privacy_engine,
            rank=rank,
        )
        return self.train_from_state(
            state, training_data, eval_data, metric_reporter, train_config
        )

    @timing.time("Trainer.train_from_state")
    def train_from_state(
        self,
        state: TrainingState,
        training_data: BatchIterator,
        eval_data: BatchIterator,
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
            train_iter (BatchIterator): batch iterator of training data
            eval_iter (BatchIterator): batch iterator of evaluation data
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

        self.sparsifier.initialize(
            self, state, eval_data, metric_reporter, train_config
        )

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
                model.eval(Stage.EVAL)
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
                model.eval(Stage.EVAL)
                print(f"start evaluating finalized state")
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

    @timing.report_snapshot
    def run_epoch(
        self, state: TrainingState, data: BatchIterator, metric_reporter: MetricReporter
    ):
        # This method is due for some refactoring, pushing it off because it interacts
        # with the metric reporter too much. Much of the logic here either changes in
        # the NewTaskTrainer or should change with a better metric reporter design.
        report_metric = state.stage != Stage.TRAIN or self.config.report_train_metrics
        model = state.model
        samples = []
        is_data_empty = True

        """
        Sometimes, a batch of inputs is too large to fit into GPU, which has to
        be split into several micro-batches. However, to improve efficiency,
        it would be helpful to only apply params/gradients sync at original batch
        boundaries instead of micro-batch boundaries.
        num_accumulated_batches specified the number of accumulating gradients
        locally before sync gradients, total training_batch_size =
        train_batch_size x num_accumulated_batches and it will improve the system
        performance by reduce the total network transfer bytes.
        """
        for sample in enumerate(data):
            is_data_empty = False
            samples.append(sample)
            if (
                state.stage != Stage.TRAIN
                or len(samples) == self.config.num_accumulated_batches
            ):
                self.run_step(samples, state, metric_reporter, report_metric)
                samples = []
        if samples:
            self.run_step(samples, state, metric_reporter, report_metric)
            samples = []

        metrics = None
        if report_metric:
            if is_data_empty:
                error_msg = (
                    f"Trying to report metric for stage {state.stage}, but no data was "
                    "found. Either disable metric reporting for this stage, pass in "
                    "non-empty data, or see if data fields are misnamed (warnings "
                    "would appear in preceding stdout logs)."
                )
                raise ValueError(error_msg)

            with timing.time("report metrics"):
                metrics = metric_reporter.report_metric(
                    model,
                    state.stage,
                    state.epoch,
                    print_to_channels=(state.rank == 0),
                    optimizer=getattr(
                        state, "optimizer", None
                    ),  # optimizer is not present during test
                    privacy_engine=getattr(state, "privacy_engine", None),
                )
        else:
            metric_reporter._reset()

        return metrics

    @timing.time("run_step")
    def run_step(
        self,
        samples: List[Any],
        state: TrainingState,
        metric_reporter: MetricReporter,
        report_metric: bool,
    ):
        sample_size = len(samples)
        assert sample_size <= self.config.num_accumulated_batches

        model = state.model
        self.zero_grads(state)
        for idx, (batch_id, (inputs, targets, context)) in enumerate(samples):
            with contextlib_ExitStack() as exit_stack:
                maybe_accumulate_gradients(exit_stack, model, idx, sample_size)
                # pass context to model to use in forward call if needed
                model.contextualize(context)
                with timing.time("model.forward"):
                    logits = model(*inputs)

                with timing.time("compute loss"):
                    loss = precision.maybe_float(
                        model.get_loss(logits, targets, context)
                    )
                    if BatchContext.IGNORE_LOSS in context:
                        loss *= 0
                    elif sample_size > 1:
                        # gradients averaged per batch and accumulated across samples.
                        # divide sample_size to let gradients averaged per example
                        loss = loss / sample_size

                self.backprop(state, loss)

            if report_metric:
                with timing.time("get pred"):
                    preds, scores = model.get_pred(
                        logits, targets, context, state.stage, *inputs
                    )

                with timing.time("add metrics"):
                    metric_reporter.add_batch_stats(
                        batch_id, preds, targets, scores, loss.item(), inputs, **context
                    )

            if batch_id % self.config.num_samples_to_log_progress == 0:
                print(
                    f"Running batch {batch_id} for epoch {state.epoch} \
                        in {state.stage} stage",
                    flush=True,
                )
        # update gradients after len(samples) forward & backward
        self.optimizer_step(state)
        with timing.time("add gradients"):
            if report_metric and state.stage == Stage.TRAIN:
                metric_reporter.add_gradients(state.model)
        self.sparsification_step(state)


class TaskTrainer(Trainer):
    __EXPANSIBLE__ = True

    class Config(Trainer.Config):
        """Make mypy happy"""

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
        for idx, (batch_id, (raw_batch, batch)) in enumerate(samples):
            with contextlib_ExitStack() as exit_stack:
                # enter ddp no_sync context and fp16 delay_scale context if needed
                maybe_accumulate_gradients(exit_stack, model, idx, sample_size)
                with timing.time("model.train_batch"):
                    loss, metric_data = model.train_batch(model, batch, state)
                    if sample_size > 1:
                        # gradients averaged per batch and accumulated across samples.
                        # divide sample_size to let gradients averaged per example
                        loss = loss / sample_size
                self.backprop(state, loss)

            if report_metric:
                with timing.time("add metrics"):
                    metric_reporter.add_batch_stats(
                        batch_id,
                        *metric_data,
                        # TODO merge this step into add_batch_stats once all data
                        # migration is done
                        **metric_reporter.batch_context(raw_batch, batch),
                    )
                if batch_id % self.config.num_samples_to_log_progress == 0:
                    metric_reporter.report_realtime_metric(state.stage)
        # update gradients after #len(samples) forward & backward
        self.optimizer_step(state)
        with timing.time("add gradients"):
            if report_metric and state.stage == Stage.TRAIN:
                metric_reporter.add_gradients(state.model)
        self.sparsification_step(state)

    def _prepare_scheduler(self, training_batches, scheduler=None):
        """Batch based schedulers require knowing the number of batches in
        the data. We're not supporting that yet with the Data api, need to figure out
        how to expose this info or restructure batch-based schedulers to not need it."""
        if scheduler.batch_based_schedulers:
            raise Exception("New tasks don't yet support batch-based scheduling")
        return scheduler
