#!/usr/bin/env python3

from typing import Optional, Union

from pytext.config import ConfigBase
from pytext.config.component import register_jobspec
from pytext.config.field_config import DocLabelConfig, LabelConfig, WordLabelConfig
from pytext.config import pair_classification as PairClassificationTask
from pytext.config.pytext_config import OptimizerParams
from pytext.data import (
    BPTTLanguageModelDataHandler,
    ContextualIntentSlotModelDataHandler,
    JointModelDataHandler,
    LanguageModelDataHandler,
    PairClassificationDataHandler,
    SeqModelDataHandler,
)
from pytext.data.featurizer import Featurizer
from pytext.exporters.exporter import TextModelExporter

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
from pytext.models.embeddings.seq_token_embedding import SequenceTokenEmbedding
from pytext.models.embeddings.token_embedding import FeatureConfig
from pytext.models.ensembles.bagging_doc_ensemble import BaggingDocEnsemble
from pytext.models.ensembles.bagging_joint_ensemble import BaggingJointEnsemble
from pytext.models.joint_model import JointModel
from pytext.models.language_models.lmlstm import LMLSTM
from pytext.models.pair_classification_model import PairClassificationModel
from pytext.models.seq_models.contextual_intent_slot import ContextualIntentSlotModel
from pytext.models.seq_models.seqnn import SeqNNModel
from pytext.models.word_model import WordTaggingModel
from pytext.optimizer.scheduler import Scheduler
from pytext.trainers import Trainer
from pytext.trainers.ensemble_trainer import EnsembleTrainer


class JobSpecBase(ConfigBase):
    features: FeatureConfig = FeatureConfig()
    optimizer: OptimizerParams = OptimizerParams()
    exporter: Optional[TextModelExporter.Config] = None
    scheduler: Optional[Scheduler.Config] = Scheduler.Config()
    featurizer: Featurizer.Config


# TODO better to have separate jobspec for different ensemble model
class EnsembleJobSpec(JobSpecBase, ConfigBase):
    model: Union[BaggingDocEnsemble.Config, BaggingJointEnsemble.Config]
    trainer: EnsembleTrainer.Config = EnsembleTrainer.Config()
    labels: LabelConfig = LabelConfig(doc_label=DocLabelConfig())
    data_handler: JointModelDataHandler.Config = JointModelDataHandler.Config()
    metric_reporter: Union[
        ClassificationMetricReporter.Config, IntentSlotMetricReporter.Config
    ]


class DocClassificationJobSpec(JobSpecBase, ConfigBase):
    model: DocModel.Config = DocModel.Config()
    trainer: Trainer.Config = Trainer.Config()
    labels: LabelConfig = LabelConfig(doc_label=DocLabelConfig())
    data_handler: JointModelDataHandler.Config = JointModelDataHandler.Config()
    metric_reporter: ClassificationMetricReporter.Config = ClassificationMetricReporter.Config()


class WordTaggingJobSpec(JobSpecBase, ConfigBase):
    model: WordTaggingModel.Config = WordTaggingModel.Config()
    trainer: Trainer.Config = Trainer.Config()
    labels: LabelConfig = LabelConfig(word_label=WordLabelConfig())
    data_handler: JointModelDataHandler.Config = JointModelDataHandler.Config()
    metric_reporter: WordTaggingMetricReporter.Config = WordTaggingMetricReporter.Config()


class JointTextJobSpec(JobSpecBase, ConfigBase):
    model: JointModel.Config = JointModel.Config()
    trainer: Trainer.Config = Trainer.Config()
    labels: LabelConfig = LabelConfig(
        doc_label=DocLabelConfig(), word_label=WordLabelConfig()
    )
    data_handler: JointModelDataHandler.Config = JointModelDataHandler.Config()
    metric_reporter: IntentSlotMetricReporter.Config = IntentSlotMetricReporter.Config()


class LMJobSpec(JobSpecBase, ConfigBase):
    data_handler: Union[
        LanguageModelDataHandler.Config, BPTTLanguageModelDataHandler.Config
    ] = LanguageModelDataHandler.Config()
    model: LMLSTM.Config
    trainer: Trainer.Config = Trainer.Config()
    labels: Optional[LabelConfig] = None
    metric_reporter: LanguageModelMetricReporter.Config = LanguageModelMetricReporter.Config()


class PairClassificationJobSpec(JobSpecBase, ConfigBase):
    features: PairClassificationTask.ModelInputConfig = PairClassificationTask.ModelInputConfig()
    model: PairClassificationModel.Config = PairClassificationModel.Config()
    data_handler: PairClassificationDataHandler.Config = PairClassificationDataHandler.Config()
    trainer: Trainer.Config = Trainer.Config()
    labels: PairClassificationTask.TargetConfig = PairClassificationTask.TargetConfig()
    metric_reporter: ClassificationMetricReporter.Config = ClassificationMetricReporter.Config()


class SeqNNJobSpec(JobSpecBase, ConfigBase):
    model: SeqNNModel.Config = SeqNNModel.Config()
    trainer: Trainer.Config = Trainer.Config()
    labels: LabelConfig = LabelConfig(doc_label=DocLabelConfig())
    data_handler: SeqModelDataHandler.Config = SeqModelDataHandler.Config()
    metric_reporter: ClassificationMetricReporter.Config = ClassificationMetricReporter.Config()


class ContextualIntentSlotJobSpec(JobSpecBase, ConfigBase):
    features: SequenceTokenEmbedding.Config = SequenceTokenEmbedding.Config()
    model: ContextualIntentSlotModel.Config = ContextualIntentSlotModel.Config()
    trainer: Trainer.Config = Trainer.Config()
    labels: LabelConfig = LabelConfig(
        doc_label=DocLabelConfig(), word_label=WordLabelConfig()
    )
    data_handler: ContextualIntentSlotModelDataHandler.Config = ContextualIntentSlotModelDataHandler.Config()
    metric_reporter: IntentSlotMetricReporter.Config = IntentSlotMetricReporter.Config()


def register_builtin_jobspecs():
    register_jobspec(
        (
            DocClassificationJobSpec,
            WordTaggingJobSpec,
            JointTextJobSpec,
            LMJobSpec,
            EnsembleJobSpec,
            PairClassificationJobSpec,
            SeqNNJobSpec,
            ContextualIntentSlotJobSpec,
        )
    )
