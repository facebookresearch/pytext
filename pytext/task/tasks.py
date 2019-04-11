#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Union

from pytext.common.constants import DatasetFieldName
from pytext.config import (
    contextual_intent_slot as ContextualIntentSlot,
    doc_classification as DocClassification,
    pair_classification as PairClassification,
    query_document_pairwise_ranking as QueryDocumentPairwiseRanking,
)
from pytext.config.field_config import DocLabelConfig, TargetConfigBase, WordLabelConfig
from pytext.data import (
    BPTTLanguageModelDataHandler,
    CompositionalDataHandler,
    ContextualIntentSlotModelDataHandler,
    DocClassificationDataHandler,
    JointModelDataHandler,
    LanguageModelDataHandler,
    PairClassificationDataHandler,
    QueryDocumentPairwiseRankingDataHandler,
    SeqModelDataHandler,
)
from pytext.data.tensorizers import Tensorizer
from pytext.exporters import DenseFeatureExporter
from pytext.metric_reporters import (
    ClassificationMetricReporter,
    CompositionalMetricReporter,
    IntentSlotMetricReporter,
    LanguageModelMetricReporter,
    MetricReporter,
    PairwiseRankingMetricReporter,
    SimpleWordTaggingMetricReporter,
    WordTaggingMetricReporter,
)
from pytext.models.doc_model import DocModel
from pytext.models.ensembles import BaggingDocEnsemble, BaggingIntentSlotEnsemble
from pytext.models.joint_model import JointModel
from pytext.models.language_models.lmlstm import LMLSTM
from pytext.models.model import Model
from pytext.models.pair_classification_model import PairClassificationModel
from pytext.models.query_document_pairwise_ranking_model import (
    QueryDocumentPairwiseRankingModel,
)
from pytext.models.semantic_parsers.rnng.rnng_parser import RNNGParser
from pytext.models.seq_models.contextual_intent_slot import ContextualIntentSlotModel
from pytext.models.seq_models.seqnn import SeqNNModel
from pytext.models.word_model import NewWordTaggingModel, WordTaggingModel
from pytext.task import Task
from pytext.task.new_task import NewTask
from pytext.trainers import EnsembleTrainer, HogwildTrainer, Trainer
from pytext.utils import distributed


class QueryDocumentPairwiseRankingTask(Task):
    class Config(Task.Config):
        features: QueryDocumentPairwiseRanking.ModelInputConfig = (
            QueryDocumentPairwiseRanking.ModelInputConfig()
        )
        model: QueryDocumentPairwiseRankingModel.Config = (
            QueryDocumentPairwiseRankingModel.Config()
        )
        data_handler: QueryDocumentPairwiseRankingDataHandler.Config = (
            QueryDocumentPairwiseRankingDataHandler.Config()
        )
        trainer: Trainer.Config = Trainer.Config()
        labels: Optional[DocLabelConfig] = None
        metric_reporter: PairwiseRankingMetricReporter.Config = (
            PairwiseRankingMetricReporter.Config()
        )


# TODO better to have separate Task for different ensemble model
class EnsembleTask(Task):
    class Config(Task.Config):
        model: Union[BaggingDocEnsemble.Config, BaggingIntentSlotEnsemble.Config]
        trainer: EnsembleTrainer.Config = EnsembleTrainer.Config()
        labels: List[TargetConfigBase]
        metric_reporter: Union[
            ClassificationMetricReporter.Config, IntentSlotMetricReporter.Config
        ] = ClassificationMetricReporter.Config()
        data_handler: JointModelDataHandler.Config = JointModelDataHandler.Config()

    def train_single_model(
        self, train_config, model_id, rank=0, world_size=1, dist_init_url=""
    ):
        assert model_id >= 0 and model_id < len(self.model.models)

        train_iter = self.data_handler.get_train_iter(rank, world_size)
        eval_iter = self.data_handler.get_eval_iter()
        if dist_init_url and world_size > 1:
            distributed.dist_init(rank, world_size, dist_init_url)

        return self.trainer.train_single_model(
            train_iter,
            eval_iter,
            self.model.models[model_id],
            self.metric_reporter,
            train_config,
        )

    @classmethod
    def example_config(cls):
        return cls.Config(
            labels=[DocLabelConfig(), WordLabelConfig()],
            model=BaggingDocEnsemble.Config(models=[DocModel.Config()]),
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
        exporter: Optional[DenseFeatureExporter.Config] = None

    @classmethod
    def format_prediction(cls, predictions, scores, context, target_meta):
        target_names = target_meta.vocab.itos
        for prediction, score in zip(predictions, scores):
            score_with_name = {n: s for n, s in zip(target_names, score.tolist())}
            yield {
                "prediction": target_names[prediction.data],
                "score": score_with_name,
            }


class WordTaggingTask(Task):
    class Config(Task.Config):
        model: WordTaggingModel.Config = WordTaggingModel.Config()
        trainer: Trainer.Config = Trainer.Config()
        labels: WordLabelConfig = WordLabelConfig()
        data_handler: JointModelDataHandler.Config = JointModelDataHandler.Config()
        metric_reporter: WordTaggingMetricReporter.Config = (
            WordTaggingMetricReporter.Config()
        )

    @classmethod
    def format_prediction(cls, predictions, scores, context, target_meta):
        target_names = target_meta.vocab.itos
        for prediction, score, token_ranges in zip(
            predictions, scores, context[DatasetFieldName.TOKEN_RANGE]
        ):
            yield [
                {
                    "prediction": target_names[word_pred.data],
                    "score": {n: s for n, s in zip(target_names, word_score.tolist())},
                    "token_range": token_range,
                }
                for word_pred, word_score, token_range in zip(
                    prediction, score, token_ranges
                )
            ]


class NewWordTaggingTask(NewTask):
    class Config(NewTask.Config):
        model: NewWordTaggingModel.Config = NewWordTaggingModel.Config()
        metric_reporter: SimpleWordTaggingMetricReporter.Config = SimpleWordTaggingMetricReporter.Config()

    @classmethod
    def create_metric_reporter(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        return SimpleWordTaggingMetricReporter.from_config_and_label_names(
            config.metric_reporter, list(tensorizers["labels"].vocab)
        )


class JointTextTask(Task):
    class Config(Task.Config):
        labels: List[TargetConfigBase]
        model: JointModel.Config = JointModel.Config()
        trainer: Trainer.Config = Trainer.Config()
        data_handler: JointModelDataHandler.Config = JointModelDataHandler.Config()
        metric_reporter: IntentSlotMetricReporter.Config = (
            IntentSlotMetricReporter.Config()
        )

    @classmethod
    def format_prediction(cls, predictions, scores, context, target_meta):
        doc_preds, word_preds = predictions
        doc_scores, word_scores = scores
        doc_target_meta, word_target_meta = target_meta

        for intent, slot in zip(
            DocClassificationTask.format_prediction(
                doc_preds, doc_scores, context, doc_target_meta
            ),
            WordTaggingTask.format_prediction(
                word_preds, word_scores, context, word_target_meta
            ),
        ):
            yield {"intent": intent, "slot": slot}

    @classmethod
    def example_config(cls):
        return cls.Config(labels=[DocLabelConfig(), WordLabelConfig()])


class LMTask(Task):
    class Config(Task.Config):
        data_handler: Union[
            LanguageModelDataHandler.Config, BPTTLanguageModelDataHandler.Config
        ] = LanguageModelDataHandler.Config()
        model: LMLSTM.Config = LMLSTM.Config()
        trainer: Trainer.Config = Trainer.Config()
        labels: Optional[WordLabelConfig] = None
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
        labels: DocLabelConfig = DocLabelConfig()
        data_handler: SeqModelDataHandler.Config = SeqModelDataHandler.Config()
        metric_reporter: ClassificationMetricReporter.Config = (
            ClassificationMetricReporter.Config()
        )


class ContextualIntentSlotTask(Task):
    class Config(Task.Config):
        labels: ContextualIntentSlot.TargetConfig
        features: ContextualIntentSlot.ModelInputConfig = (
            ContextualIntentSlot.ModelInputConfig()
        )
        model: ContextualIntentSlotModel.Config = ContextualIntentSlotModel.Config()
        trainer: Trainer.Config = Trainer.Config()
        data_handler: ContextualIntentSlotModelDataHandler.Config = (
            ContextualIntentSlotModelDataHandler.Config()
        )
        metric_reporter: IntentSlotMetricReporter.Config = (
            IntentSlotMetricReporter.Config()
        )

    @classmethod
    def example_config(cls):
        return cls.Config(labels=[DocLabelConfig(), WordLabelConfig()])


class SemanticParsingTask(Task):
    class Config(Task.Config):
        model: RNNGParser.Config = RNNGParser.Config()
        trainer: HogwildTrainer.Config = HogwildTrainer.Config()
        data_handler: CompositionalDataHandler.Config = CompositionalDataHandler.Config()
        labels: Optional[WordLabelConfig] = None
        metric_reporter: CompositionalMetricReporter.Config = CompositionalMetricReporter.Config()
