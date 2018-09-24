#!/usr/bin/env python3
from typing import Optional, Union

from pytext.config import ConfigBase
from pytext.config.component import register_jobspec
from pytext.config.field_config import DocLabelConfig, LabelConfig, WordLabelConfig
from pytext.config.pytext_config import OptimizerParams, SchedulerParams
from pytext.data.bptt_lm_data_handler import BPTTLanguageModelDataHandler
from pytext.data.compositional_data_handler import CompositionalDataHandler
from pytext.data.joint_data_handler import JointModelDataHandler
from pytext.data.language_model_data_handler import LanguageModelDataHandler
from pytext.exporters.exporter import TextModelExporter
from pytext.fb.rnng.config import CompositionalTrainerConfig, RNNGConfig, Seq2SeqConfig
from pytext.metric_reporters.classification_metric_reporter import (
    ClassificationMetricReporter
)
from pytext.metric_reporters.intent_slot_detection_metric_reporter import (
    IntentSlotMetricReporter
)
from pytext.metric_reporters.language_model_metric_reporter import (
    LanguageModelMetricReporter
)
from pytext.metric_reporters.word_tagging_metric_reporter import (
    WordTaggingMetricReporter
)
from pytext.models.doc_model import DocModel
from pytext.models.embeddings.token_embedding import FeatureConfig
from pytext.models.ensembles.bagging_doc_ensemble import BaggingDocEnsemble
from pytext.models.ensembles.bagging_joint_ensemble import BaggingJointEnsemble
from pytext.models.joint_model import JointModel
from pytext.models.language_models.lmlstm import LMLSTM
from pytext.models.word_model import WordTaggingModel
from pytext.trainers import Trainer
from pytext.trainers.ensemble_trainer import EnsembleTrainer


class JobSpecBase(ConfigBase):
    features: FeatureConfig = FeatureConfig()
    optimizer: OptimizerParams = OptimizerParams()
    exporter: Optional[TextModelExporter.Config] = None
    scheduler: Optional[SchedulerParams] = SchedulerParams()


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
    ]
    model: LMLSTM.Config
    trainer: Trainer.Config = Trainer.Config()
    labels: Optional[LabelConfig] = None
    metric_reporter: LanguageModelMetricReporter.Config = LanguageModelMetricReporter.Config()


class SemanticParsingJobSpec(JobSpecBase, ConfigBase):
    model: Union[RNNGConfig, Seq2SeqConfig]
    trainer: CompositionalTrainerConfig = CompositionalTrainerConfig()
    data_handler: CompositionalDataHandler.Config = (CompositionalDataHandler.Config())


def register_builtin_jobspecs():
    register_jobspec(
        (
            DocClassificationJobSpec,
            WordTaggingJobSpec,
            JointTextJobSpec,
            LMJobSpec,
            SemanticParsingJobSpec,
            EnsembleJobSpec,
        )
    )
