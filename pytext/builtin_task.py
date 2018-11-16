#!/usr/bin/env python3

from typing import Optional, Union

from pytext.config import (
    contextual_intent_slot as ContextualIntentSlot,
    doc_classification as DocClassification,
    pair_classification as PairClassification,
)
from pytext.config.component import register_tasks
from pytext.config.field_config import DocLabelConfig, LabelConfig, WordLabelConfig
from pytext.data import (
    BPTTLanguageModelDataHandler,
    ContextualIntentSlotModelDataHandler,
    DocClassificationDataHandler,
    JointModelDataHandler,
    LanguageModelDataHandler,
    PairClassificationDataHandler,
    SeqModelDataHandler,
)
from pytext.metric_reporters.classification_metric_reporter import (
    ClassificationMetricReporter,
)
from pytext.metric_reporters.intent_slot_detection_metric_reporter import (
    IntentSlotMetricReporter,
)
from pytext.metric_reporters.language_model_metric_reporter import (
    LanguageModelMetricReporter,
)
from pytext.metric_reporters.word_tagging_metric_reporter import (
    WordTaggingMetricReporter,
)
from pytext.models.doc_model import DocModel
from pytext.models.ensembles.bagging_doc_ensemble import BaggingDocEnsemble
from pytext.models.ensembles.bagging_joint_ensemble import BaggingJointEnsemble
from pytext.models.joint_model import JointModel
from pytext.models.language_models.lmlstm import LMLSTM
from pytext.models.pair_classification_model import PairClassificationModel
from pytext.models.seq_models.contextual_intent_slot import ContextualIntentSlotModel
from pytext.models.seq_models.seqnn import SeqNNModel
from pytext.models.word_model import WordTaggingModel
from pytext.task import Task
from pytext.trainers import Trainer
from pytext.trainers.ensemble_trainer import EnsembleTrainer


# TODO better to have separate Task for different ensemble model
class EnsembleTask(Task):
    class Config(Task.Config):
        model: Union[BaggingDocEnsemble.Config, BaggingJointEnsemble.Config]
        trainer: EnsembleTrainer.Config = EnsembleTrainer.Config()
        labels: LabelConfig = LabelConfig(doc_label=DocLabelConfig())
        data_handler: JointModelDataHandler.Config = JointModelDataHandler.Config()
        metric_reporter: Union[
            ClassificationMetricReporter.Config, IntentSlotMetricReporter.Config
        ]

    def train_single_model(self, train_config, model_id, rank=0, world_size=1):
        assert model_id >= 0 and model_id < len(self.model.models)
        return self.trainer.train_single_model(
            self.data_handler.get_train_iter(rank, world_size),
            self.data_handler.get_eval_iter(),
            self.model.models[model_id],
            self.metric_reporter,
            train_config,
            self.optimizers,
            self.lr_scheduler,
        )


class DocClassificationTask(Task):
    class Config(Task.Config):
        model: DocModel.Config = DocModel.Config()
        trainer: Trainer.Config = Trainer.Config()
        features: DocClassification.ModelInputConfig = (
            DocClassification.ModelInputConfig()
        )
        labels: DocClassification.TargetConfig = DocClassification.TargetConfig()
        data_handler: DocClassificationDataHandler.Config = (
            DocClassificationDataHandler.Config()
        )
        metric_reporter: ClassificationMetricReporter.Config = (
            ClassificationMetricReporter.Config()
        )


class WordTaggingTask(Task):
    class Config(Task.Config):
        model: WordTaggingModel.Config = WordTaggingModel.Config()
        trainer: Trainer.Config = Trainer.Config()
        labels: LabelConfig = LabelConfig(word_label=WordLabelConfig())
        data_handler: JointModelDataHandler.Config = JointModelDataHandler.Config()
        metric_reporter: WordTaggingMetricReporter.Config = (
            WordTaggingMetricReporter.Config()
        )


class JointTextTask(Task):
    class Config(Task.Config):
        model: JointModel.Config = JointModel.Config()
        trainer: Trainer.Config = Trainer.Config()
        labels: LabelConfig = LabelConfig(
            doc_label=DocLabelConfig(), word_label=WordLabelConfig()
        )
        data_handler: JointModelDataHandler.Config = JointModelDataHandler.Config()
        metric_reporter: IntentSlotMetricReporter.Config = (
            IntentSlotMetricReporter.Config()
        )


class LMTask(Task):
    class Config(Task.Config):
        data_handler: Union[
            LanguageModelDataHandler.Config, BPTTLanguageModelDataHandler.Config
        ] = LanguageModelDataHandler.Config()
        model: LMLSTM.Config
        trainer: Trainer.Config = Trainer.Config()
        labels: Optional[LabelConfig] = None
        metric_reporter: LanguageModelMetricReporter.Config = (
            LanguageModelMetricReporter.Config()
        )


class PairClassificationTask(Task):
    class Config(Task.Config):
        features: PairClassification.ModelInputConfig = (
            PairClassification.ModelInputConfig()
        )
        model: PairClassificationModel.Config = PairClassificationModel.Config()
        data_handler: PairClassificationDataHandler.Config = (
            PairClassificationDataHandler.Config()
        )
        trainer: Trainer.Config = Trainer.Config()
        labels: PairClassification.TargetConfig = PairClassification.TargetConfig()
        metric_reporter: ClassificationMetricReporter.Config = (
            ClassificationMetricReporter.Config()
        )


class SeqNNTask(Task):
    class Config(Task.Config):
        model: SeqNNModel.Config = SeqNNModel.Config()
        trainer: Trainer.Config = Trainer.Config()
        labels: LabelConfig = LabelConfig(doc_label=DocLabelConfig())
        data_handler: SeqModelDataHandler.Config = SeqModelDataHandler.Config()
        metric_reporter: ClassificationMetricReporter.Config = (
            ClassificationMetricReporter.Config()
        )


class ContextualIntentSlotTask(Task):
    class Config(Task.Config):
        features: ContextualIntentSlot.ModelInputConfig = (
            ContextualIntentSlot.ModelInputConfig()
        )
        model: ContextualIntentSlotModel.Config = ContextualIntentSlotModel.Config()
        trainer: Trainer.Config = Trainer.Config()
        labels: ContextualIntentSlot.TargetConfig = (
            ContextualIntentSlot.TargetConfig()
        )
        data_handler: ContextualIntentSlotModelDataHandler.Config = (
            ContextualIntentSlotModelDataHandler.Config()
        )
        metric_reporter: IntentSlotMetricReporter.Config = (
            IntentSlotMetricReporter.Config()
        )


def register_builtin_tasks():
    register_tasks(
        (
            DocClassificationTask,
            WordTaggingTask,
            JointTextTask,
            LMTask,
            EnsembleTask,
            PairClassification,
            SeqNNTask,
            ContextualIntentSlotTask,
        )
    )
