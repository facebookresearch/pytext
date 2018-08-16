#!/usr/bin/env python3
from typing import Optional, Union

from pytext.common.registry import register_jobspec
from pytext.config import ConfigBase
from pytext.config.field_config import (
    DocLabelConfig,
    FeatureConfig,
    LabelConfig,
    WordLabelConfig,
)
from pytext.config.pytext_config import OptimizerParams
from pytext.data.compositional_data_handler import CompositionalDataHandlerConfig
from pytext.data.joint_data_handler import JointTextModelDataHandlerConfig
from pytext.data.language_model_data_handler import LanguageModelDataHandlerConfig
from pytext.exporters.exporter import TextModelExporterConfig
from pytext.loss.classifier_loss import (
    BinaryCrossEntropyLossConfig,
    CrossEntropyLossConfig,
)
from pytext.loss.joint_loss import JointLossConfig
from pytext.loss.language_model_loss import LanguageModelCrossEntropyLossConfig
from pytext.loss.tagger_loss import CRFLossConfig, TaggerCrossEntropyLossConfig
from pytext.models.doc_models import DocBLSTMConfig, DocNNConfig
from pytext.models.ensembles.bagging_doc_ensemble import BaggingDocEnsembleConfig
from pytext.models.ensembles.bagging_joint_ensemble import BaggingJointEnsembleConfig
from pytext.models.joint_models import JointBLSTMConfig, JointCNNConfig
from pytext.models.language_models.lmlstm import LMLSTMConfig
from pytext.models.word_models import WordBLSTMConfig, WordCNNConfig
from pytext.rnng.config import CompositionalTrainerConfig, RNNGConfig, Seq2SeqConfig
from pytext.trainers.classifier_trainer import ClassifierTrainerConfig
from pytext.trainers.ensemble_trainer import EnsembleTrainerConfig
from pytext.trainers.joint_trainer import JointTrainerConfig
from pytext.trainers.language_model_trainer import LMTrainerConfig
from pytext.trainers.tagger_trainer import TaggerTrainerConfig


class EnsembleJobSpec(ConfigBase):
    model: Union[BaggingDocEnsembleConfig, BaggingJointEnsembleConfig]
    loss: Union[CrossEntropyLossConfig, BinaryCrossEntropyLossConfig, JointLossConfig]
    trainer: EnsembleTrainerConfig
    data_handler: JointTextModelDataHandlerConfig = JointTextModelDataHandlerConfig()
    features: FeatureConfig = FeatureConfig()
    labels: LabelConfig = LabelConfig(doc_label=DocLabelConfig())
    optimizer: OptimizerParams = OptimizerParams()
    exporter: Optional[TextModelExporterConfig] = None


class DocClassifyJobSpec(ConfigBase):
    model: Union[DocNNConfig, DocBLSTMConfig]
    loss: Union[CrossEntropyLossConfig, BinaryCrossEntropyLossConfig]
    trainer: ClassifierTrainerConfig = ClassifierTrainerConfig()
    data_handler: JointTextModelDataHandlerConfig = JointTextModelDataHandlerConfig()
    features: FeatureConfig = FeatureConfig()
    labels: LabelConfig = LabelConfig(doc_label=DocLabelConfig())
    optimizer: OptimizerParams = OptimizerParams()
    exporter: Optional[TextModelExporterConfig] = None


class WordTagJobSpec(ConfigBase):
    model: Union[WordBLSTMConfig, WordCNNConfig]
    loss: Union[CRFLossConfig, TaggerCrossEntropyLossConfig]
    trainer: TaggerTrainerConfig = TaggerTrainerConfig()
    data_handler: JointTextModelDataHandlerConfig = JointTextModelDataHandlerConfig()
    features: FeatureConfig = FeatureConfig()
    labels: LabelConfig = LabelConfig(word_label=WordLabelConfig())
    optimizer: OptimizerParams = OptimizerParams()
    exporter: Optional[TextModelExporterConfig] = None


class JointTextJobSpec(ConfigBase):
    model: Union[JointBLSTMConfig, JointCNNConfig]
    loss: JointLossConfig
    trainer: JointTrainerConfig = JointTrainerConfig()
    data_handler: JointTextModelDataHandlerConfig = JointTextModelDataHandlerConfig()
    features: FeatureConfig = FeatureConfig()
    labels: LabelConfig = LabelConfig(doc_label=DocLabelConfig(),word_label=WordLabelConfig())
    optimizer: OptimizerParams = OptimizerParams()
    exporter: Optional[TextModelExporterConfig] = None


class LMJobSpec(ConfigBase):
    model: LMLSTMConfig = LMLSTMConfig()
    loss: LanguageModelCrossEntropyLossConfig = LanguageModelCrossEntropyLossConfig()
    trainer: LMTrainerConfig = LMTrainerConfig()
    data_handler: LanguageModelDataHandlerConfig = LanguageModelDataHandlerConfig()
    features: FeatureConfig = FeatureConfig()
    labels: Optional[LabelConfig] = None
    optimizer: OptimizerParams = OptimizerParams()
    # TODO implement the actual exporter later
    exporter: Optional[TextModelExporterConfig] = None


class SemanticParsingJobSpec(ConfigBase):
    model: Union[RNNGConfig, Seq2SeqConfig]
    trainer: CompositionalTrainerConfig = CompositionalTrainerConfig()
    data_handler: CompositionalDataHandlerConfig = CompositionalDataHandlerConfig()
    features: FeatureConfig = FeatureConfig()
    optimizer: OptimizerParams = OptimizerParams()
    # TODO implement the actual exporter later
    exporter: Optional[TextModelExporterConfig] = None


def register_builtin_jobspecs():
    register_jobspec(
        [
            DocClassifyJobSpec,
            WordTagJobSpec,
            JointTextJobSpec,
            LMJobSpec,
            SemanticParsingJobSpec,
            EnsembleJobSpec,
        ]
    )
