#!/usr/bin/env python3

import copy
from typing import Dict, Type

import torch.nn as nn
from pytext.common.constants import DocWordModelType
from pytext.config.ttypes import Model, PyTextConfig
from pytext.data.data_handler import DataHandler
from pytext.data.joint_data_handler import JointModelDataHandler
from pytext.data.language_model_data_handler import LanguageModelDataHandler
from pytext.experimental.data.i18n_joint_data_handler import (
    I18NJointModelDataHandler
)
from pytext.experimental.models.i18n_jointblstm import I18NJointBLSTM
from pytext.exporters.exporter import TextModelExporter
from pytext.loss.classifier_loss import get_classifier_loss
from pytext.loss.joint_loss import JointLoss
from pytext.loss.language_model_loss import get_language_model_loss
from pytext.loss.tagger_loss import get_tagger_loss
from pytext.models.doc_models.docblstm import DocBLSTM
from pytext.models.doc_models.docnn import DocNN
from pytext.models.ensembles.bagging_doc_ensemble import BaggingDocEnsemble
from pytext.models.ensembles.bagging_joint_ensemble import (
    BaggingJointEnsemble
)
from pytext.models.ensembles.ensemble import Ensemble
from pytext.models.joint_models.jointblstm import JointBLSTM
from pytext.models.joint_models.jointcnn import JointCNN
from pytext.models.language_models.lmlstm import LMLSTM
from pytext.models.model import Model as ModelCls
from pytext.models.word_models.wordblstm import WordBLSTM
from pytext.models.word_models.wordcnn import WordCNN
from pytext.predictors.classifier_predictor import ClassifierPredictor
from pytext.predictors.joint_predictor import JointPredictor
from pytext.predictors.tagger_predictor import TaggerPredictor
from pytext.trainers.classifier_trainer import ClassifierTrainer
from pytext.trainers.ensemble_trainer import EnsembleTrainer
from pytext.trainers.joint_trainer import JointTrainer
from pytext.trainers.language_model_trainer import LanguageModelTrainer
from pytext.trainers.tagger_trainer import TaggerTrainer

from .component import ComponentCreator


class Registry:
    _content: Dict = {}

    @staticmethod
    def add(model: int, creator: ComponentCreator):
        if model in Registry._content:
            raise Exception("model {} already registered".format(model))
        Registry._content[model] = creator

    @staticmethod
    def get(model: int) -> ComponentCreator:
        if model not in Registry._content:
            raise Exception("unregistered model type {}".format(model))
        return Registry._content[model]


class EnsembleComponentCreator(ComponentCreator):
    def __init__(self, model_cls: Type[Ensemble]) -> None:
        self.model_cls = model_cls

    def _get_sub_model_creator(self, config: PyTextConfig):
        # use the first model config
        return Registry.get(config.model.value.models[0].getType())

    def create_data_handler(self, config: PyTextConfig, **kwargs):
        return self._get_sub_model_creator(config).create_data_handler(config, **kwargs)

    def create_trainer(self, config: PyTextConfig, **metadata):
        return EnsembleTrainer(
            self._get_sub_model_creator(config).create_trainer(config, **metadata)
        )

    def create_model(self, config: PyTextConfig, **metadata):
        n_models = len(config.model.value.models)
        models = []
        config_new = copy.deepcopy(config)
        for i in range(n_models):
            config_new.model = config.model.value.models[i]
            models.append(
                Registry.get(config_new.model.getType()).create_model(
                    config_new, **metadata
                )
            )
        return self.model_cls(config, models, **metadata)

    def create_predictor(self, config: PyTextConfig, **metadata):
        return self._get_sub_model_creator(config).create_predictor(config, **metadata)

    def create_exporter(self, config: PyTextConfig, **metadata):
        return self._get_sub_model_creator(config).create_exporter(config, **metadata)

    def create_loss(self, config: PyTextConfig, **metadata):
        return self._get_sub_model_creator(config).create_loss(config, **metadata)


class TextModelComponentCreator(ComponentCreator):
    def __init__(
        self,
        model_type: DocWordModelType,
        model_cls: Type[ModelCls],
        data_handler_cls: Type[DataHandler] = JointModelDataHandler,
    ) -> None:
        self.model_type = model_type
        self.model_cls = model_cls
        self.data_handler_cls = data_handler_cls

    def create_data_handler(self, config: PyTextConfig, **kwargs):
        return self.data_handler_cls.from_config(config, self.model_type)

    def create_trainer(self, config: PyTextConfig, **metadata):
        loss = self.create_loss(config, **metadata)
        if self.model_type == DocWordModelType.DOC:
            return ClassifierTrainer(loss)
        elif self.model_type == DocWordModelType.WORD:
            return TaggerTrainer(loss)
        else:
            return JointTrainer(loss)

    def create_model(self, config: PyTextConfig, **metadata):
        return self.model_cls.from_config(config, **metadata)

    def create_predictor(
        self,
        config: PyTextConfig,
        model: nn.Module,
        data_handler: DataHandler,
        **metadata
    ):
        if self.model_type == DocWordModelType.DOC:
            return ClassifierPredictor(model, data_handler)
        elif self.model_type == DocWordModelType.WORD:
            return TaggerPredictor(model, data_handler)
        else:
            return JointPredictor(model, data_handler)

    def create_exporter(self, config: PyTextConfig, **metadata):
        return TextModelExporter.from_config(config, self.model_type, **metadata)

    def create_loss(self, config: PyTextConfig, **metadata):
        if self.model_type == DocWordModelType.DOC:
            return get_classifier_loss(config)
        elif self.model_type == DocWordModelType.WORD:
            return get_tagger_loss(config)
        else:
            return JointLoss(get_classifier_loss(config), get_tagger_loss(config))


class LanguageModelComponentCreator(ComponentCreator):
    def __init__(self, model_cls: Type[ModelCls]) -> None:
        self.model_cls = model_cls

    def create_data_handler(self, config: PyTextConfig, *args, **kwargs):
        return LanguageModelDataHandler.from_config(config)

    def create_model(self, config: PyTextConfig, **metadata):
        return self.model_cls.from_config(config, **metadata)

    def create_loss(self, config: PyTextConfig, **metadata):
        return get_language_model_loss(config, metadata.get("pad_idx"))

    def create_trainer(self, config: PyTextConfig, **metadata):
        loss = self.create_loss(config, **metadata)
        return LanguageModelTrainer(loss)

    def create_predictor(
        self,
        config: PyTextConfig,
        model: nn.Module,
        data_handler: DataHandler,
        **metadata
    ):
        raise NotImplementedError()

    def create_exporter(self, config: PyTextConfig, **metadata):
        raise NotImplementedError()


Registry.add(Model.DOCNN, TextModelComponentCreator(DocWordModelType.DOC, DocNN))
Registry.add(Model.DOCBLSTM, TextModelComponentCreator(DocWordModelType.DOC, DocBLSTM))
Registry.add(
    Model.WORDBLSTM, TextModelComponentCreator(DocWordModelType.WORD, WordBLSTM)
)
Registry.add(Model.WORDCNN, TextModelComponentCreator(DocWordModelType.WORD, WordCNN))
Registry.add(
    Model.JOINTBLSTM, TextModelComponentCreator(DocWordModelType.JOINT, JointBLSTM)
)
Registry.add(
    Model.JOINTCNN, TextModelComponentCreator(DocWordModelType.JOINT, JointCNN)
)
Registry.add(Model.LMLSTM, LanguageModelComponentCreator(LMLSTM))
# ensembles
Registry.add(Model.BAGGING_DOC_ENSEMBLE, EnsembleComponentCreator(BaggingDocEnsemble))

Registry.add(
    Model.BAGGING_JOINT_ENSEMBLE, EnsembleComponentCreator(BaggingJointEnsemble)
)

Registry.add(
    Model.I18N_JOINTBLSTM,
    TextModelComponentCreator(
        DocWordModelType.JOINT, I18NJointBLSTM, I18NJointModelDataHandler
    ),
)
