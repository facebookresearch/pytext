#!/usr/bin/env python3
from typing import Optional, Union

from pytext.config.component import register_jobspec
from pytext.config import ConfigBase
from pytext.config.field_config import (
    DocLabelConfig,
    FeatureConfig,
    LabelConfig,
    WordLabelConfig,
)
from pytext.config.pytext_config import OptimizerParams
from pytext.data.compositional_data_handler import CompositionalDataHandler
from pytext.data.joint_data_handler import JointModelDataHandler
from pytext.data.language_model_data_handler import LanguageModelDataHandler
from pytext.exporters.exporter import TextModelExporter
from pytext.loss.classifier_loss import (
    BinaryCrossEntropyLoss,
    CrossEntropyLoss,
)
from pytext.loss.joint_loss import JointLoss
from pytext.loss.language_model_loss import LanguageModelCrossEntropyLoss
from pytext.loss.tagger_loss import CRFLoss, TaggerCrossEntropyLoss
from pytext.models.doc_models import DocBLSTM, DocNN
from pytext.models.ensembles.bagging_doc_ensemble import (
    BaggingDocEnsemble
)
from pytext.models.ensembles.bagging_joint_ensemble import (
    BaggingJointEnsemble
)
from pytext.models.joint_models import JointBLSTM, JointCNN
from pytext.models.language_models.lmlstm import LMLSTM
from pytext.models.word_models import WordBLSTM, WordCNN
from pytext.fb.rnng.config import (
    CompositionalTrainerConfig,
    RNNGConfig,
    Seq2SeqConfig,
)
from pytext.trainers.classifier_trainer import ClassifierTrainer
from pytext.trainers.ensemble_trainer import EnsembleTrainer
from pytext.trainers.joint_trainer import JointTrainer
from pytext.trainers.tagger_trainer import TaggerTrainer
from pytext.trainers.language_model_trainer import LanguageModelTrainer


class JobSpecBase(ConfigBase):
    data_handler: JointModelDataHandler.Config = JointModelDataHandler.Config()
    features: FeatureConfig = FeatureConfig()
    optimizer: OptimizerParams = OptimizerParams()
    exporter: Optional[TextModelExporter.Config] = None


class EnsembleJobSpec(JobSpecBase, ConfigBase):
    model: Union[BaggingDocEnsemble.Config, BaggingJointEnsemble.Config]
    loss: Union[
        CrossEntropyLoss.Config,
        BinaryCrossEntropyLoss.Config,
        JointLoss.Config,
    ]
    trainer: EnsembleTrainer.Config
    labels: LabelConfig = LabelConfig(doc_label=DocLabelConfig())


class DocClassifyJobSpec(JobSpecBase, ConfigBase):
    model: Union[DocNN.Config, DocBLSTM.Config]
    loss: Union[CrossEntropyLoss.Config, BinaryCrossEntropyLoss.Config]
    trainer: ClassifierTrainer.Config = ClassifierTrainer.Config()
    labels: LabelConfig = LabelConfig(doc_label=DocLabelConfig())


class WordTagJobSpec(JobSpecBase, ConfigBase):
    model: Union[WordBLSTM.Config, WordCNN.Config]
    loss: Union[CRFLoss.Config, TaggerCrossEntropyLoss.Config]
    trainer: TaggerTrainer.Config = TaggerTrainer.Config()
    labels: LabelConfig = LabelConfig(word_label=WordLabelConfig())


class JointTextJobSpec(JobSpecBase, ConfigBase):
    model: Union[JointBLSTM.Config, JointCNN.Config]
    loss: JointLoss.Config
    trainer: JointTrainer.Config = JointTrainer.Config()
    labels: LabelConfig = LabelConfig(
        doc_label=DocLabelConfig(),
        word_label=WordLabelConfig(),
    )


class LMJobSpec(JobSpecBase, ConfigBase):
    model: LMLSTM.Config
    loss: LanguageModelCrossEntropyLoss.Config = (
        LanguageModelCrossEntropyLoss.Config()
    )
    trainer: LanguageModelTrainer.Config = LanguageModelTrainer.Config()
    data_handler: LanguageModelDataHandler.Config = (
        LanguageModelDataHandler.Config()
    )
    labels: Optional[LabelConfig] = None


class SemanticParsingJobSpec(JobSpecBase, ConfigBase):
    model: Union[RNNGConfig, Seq2SeqConfig]
    trainer: CompositionalTrainerConfig = CompositionalTrainerConfig()
    data_handler: CompositionalDataHandler.Config = (
        CompositionalDataHandler.Config()
    )


def register_builtin_jobspecs():
    register_jobspec((
        DocClassifyJobSpec,
        WordTagJobSpec,
        JointTextJobSpec,
        LMJobSpec,
        SemanticParsingJobSpec,
        EnsembleJobSpec,
    ))
