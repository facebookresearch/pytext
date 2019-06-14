#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Union

from pytext.common.constants import DatasetFieldName, Stage
from pytext.config import (
    contextual_intent_slot as ContextualIntentSlot,
    doc_classification as DocClassification,
    query_document_pairwise_ranking as QueryDocumentPairwiseRanking,
)
from pytext.config.field_config import DocLabelConfig, TargetConfigBase, WordLabelConfig
from pytext.config.pytext_config import PlaceHolder
from pytext.data import (
    CompositionalDataHandler,
    ContextualIntentSlotModelDataHandler,
    DocClassificationDataHandler,
    JointModelDataHandler,
    LanguageModelDataHandler,
    QueryDocumentPairwiseRankingDataHandler,
    SeqModelDataHandler,
)
from pytext.data.data import Data
from pytext.data.tensorizers import Tensorizer
from pytext.exporters import DenseFeatureExporter
from pytext.metric_reporters import (
    ClassificationMetricReporter,
    CompositionalMetricReporter,
    IntentSlotMetricReporter,
    LanguageModelMetricReporter,
    PairwiseRankingMetricReporter,
    RegressionMetricReporter,
    SequenceTaggingMetricReporter,
    WordTaggingMetricReporter,
)
from pytext.models.doc_model import DocModel, DocModel_Deprecated, DocRegressionModel
from pytext.models.ensembles import (
    BaggingDocEnsemble_Deprecated,
    BaggingDocEnsembleModel,
    BaggingIntentSlotEnsemble_Deprecated,
    EnsembleModel,
)
from pytext.models.joint_model import IntentSlotModel, JointModel
from pytext.models.language_models.lmlstm import LMLSTM, LMLSTM_Deprecated
from pytext.models.model import BaseModel
from pytext.models.pair_classification_model import BasePairwiseModel, PairwiseModel
from pytext.models.query_document_pairwise_ranking_model import (
    QueryDocPairwiseRankingModel,
    QueryDocumentPairwiseRankingModel_Deprecated,
)
from pytext.models.semantic_parsers.rnng.rnng_parser import (
    RNNGParser,
    RNNGParser_Deprecated,
)
from pytext.models.seq_models.contextual_intent_slot import ContextualIntentSlotModel
from pytext.models.seq_models.seqnn import SeqNNModel, SeqNNModel_Deprecated
from pytext.models.word_model import WordTaggingModel, WordTaggingModel_Deprecated
from pytext.task import Task_Deprecated
from pytext.task.new_task import NewTask
from pytext.trainers import (
    EnsembleTrainer,
    EnsembleTrainer_Deprecated,
    HogwildTrainer,
    HogwildTrainer_Deprecated,
    Trainer,
)


class QueryDocumentPairwiseRankingTask_Deprecated(Task_Deprecated):
    class Config(Task_Deprecated.Config):
        features: QueryDocumentPairwiseRanking.ModelInputConfig = (
            QueryDocumentPairwiseRanking.ModelInputConfig()
        )
        model: QueryDocumentPairwiseRankingModel_Deprecated.Config = (
            QueryDocumentPairwiseRankingModel_Deprecated.Config()
        )
        data_handler: QueryDocumentPairwiseRankingDataHandler.Config = (
            QueryDocumentPairwiseRankingDataHandler.Config()
        )
        trainer: Trainer.Config = Trainer.Config()
        labels: Optional[DocLabelConfig] = None
        metric_reporter: PairwiseRankingMetricReporter.Config = (
            PairwiseRankingMetricReporter.Config()
        )


class QueryDocumentPairwiseRankingTask(NewTask):
    class Config(NewTask.Config):
        model: QueryDocPairwiseRankingModel.Config = (
            QueryDocPairwiseRankingModel.Config()
        )
        metric_reporter: PairwiseRankingMetricReporter.Config = (
            PairwiseRankingMetricReporter.Config()
        )


# TODO better to have separate Task for different ensemble model
class EnsembleTask_Deprecated(Task_Deprecated):
    class Config(Task_Deprecated.Config):
        model: Union[
            BaggingDocEnsemble_Deprecated.Config,
            BaggingIntentSlotEnsemble_Deprecated.Config,
        ]
        trainer: EnsembleTrainer_Deprecated.Config = EnsembleTrainer_Deprecated.Config()
        labels: List[TargetConfigBase]
        metric_reporter: Union[
            ClassificationMetricReporter.Config, IntentSlotMetricReporter.Config
        ] = ClassificationMetricReporter.Config()
        data_handler: JointModelDataHandler.Config = JointModelDataHandler.Config()

    def train_single_model(self, train_config, model_id, rank=0, world_size=1):
        assert model_id >= 0 and model_id < len(self.model.models)
        return self.trainer.train_single_model(
            self.data_handler.get_train_iter(rank, world_size),
            self.data_handler.get_eval_iter(),
            self.model.models[model_id],
            self.metric_reporter,
            train_config,
        )

    @classmethod
    def example_config(cls):
        return cls.Config(
            labels=[DocLabelConfig(), WordLabelConfig()],
            model=BaggingDocEnsemble_Deprecated.Config(
                models=[DocModel_Deprecated.Config()]
            ),
        )


class EnsembleTask(NewTask):
    class Config(NewTask.Config):
        model: EnsembleModel.Config
        trainer: EnsembleTrainer.Config = EnsembleTrainer.Config()
        metric_reporter: Union[
            ClassificationMetricReporter.Config, IntentSlotMetricReporter.Config
        ] = ClassificationMetricReporter.Config()

    def train_single_model(self, train_config, model_id, rank=0, world_size=1):
        return self.trainer.train_single_model(
            self.data.batches(Stage.TRAIN),
            self.data.batches(Stage.EVAL),
            self.model.models[model_id],
            self.metric_reporter,
            train_config,
        )

    @classmethod
    def example_config(cls):
        return cls.Config(
            model=BaggingDocEnsembleModel.Config(models=[DocModel.Config()])
        )


class DocClassificationTask_Deprecated(Task_Deprecated):
    class Config(Task_Deprecated.Config):
        model: DocModel_Deprecated.Config = DocModel_Deprecated.Config()
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


class DocumentClassificationTask(NewTask):
    class Config(NewTask.Config):
        model: BaseModel.Config = DocModel.Config()
        metric_reporter: ClassificationMetricReporter.Config = (
            ClassificationMetricReporter.Config()
        )


class DocumentRegressionTask(NewTask):
    class Config(NewTask.Config):
        model: DocRegressionModel.Config = DocRegressionModel.Config()
        metric_reporter: RegressionMetricReporter.Config = (
            RegressionMetricReporter.Config()
        )


class WordTaggingTask_Deprecated(Task_Deprecated):
    class Config(Task_Deprecated.Config):
        model: WordTaggingModel_Deprecated.Config = WordTaggingModel_Deprecated.Config()
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


class WordTaggingTask(NewTask):
    class Config(NewTask.Config):
        model: WordTaggingModel.Config = WordTaggingModel.Config()
        metric_reporter: SequenceTaggingMetricReporter.Config = SequenceTaggingMetricReporter.Config()

    @classmethod
    def create_metric_reporter(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        return SequenceTaggingMetricReporter.from_config(
            config.metric_reporter, tensorizers["labels"]
        )


class JointTextTask_Deprecated(Task_Deprecated):
    class Config(Task_Deprecated.Config):
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
            DocClassificationTask_Deprecated.format_prediction(
                doc_preds, doc_scores, context, doc_target_meta
            ),
            WordTaggingTask_Deprecated.format_prediction(
                word_preds, word_scores, context, word_target_meta
            ),
        ):
            yield {"intent": intent, "slot": slot}

    @classmethod
    def example_config(cls):
        return cls.Config(labels=[DocLabelConfig(), WordLabelConfig()])


class IntentSlotTask(NewTask):
    class Config(NewTask.Config):
        model: IntentSlotModel.Config = IntentSlotModel.Config()
        metric_reporter: IntentSlotMetricReporter.Config = (
            IntentSlotMetricReporter.Config()
        )


class LMTask_Deprecated(Task_Deprecated):
    class Config(Task_Deprecated.Config):
        # Have PlaceHolder to keep it as Union so we don't have to write config adapter
        # for it, this class should be removed soon
        data_handler: Union[
            LanguageModelDataHandler.Config, PlaceHolder
        ] = LanguageModelDataHandler.Config()
        model: LMLSTM_Deprecated.Config = LMLSTM_Deprecated.Config()
        trainer: Trainer.Config = Trainer.Config()
        labels: Optional[WordLabelConfig] = None
        metric_reporter: LanguageModelMetricReporter.Config = (
            LanguageModelMetricReporter.Config()
        )


class LMTask(NewTask):
    class Config(NewTask.Config):
        model: LMLSTM.Config = LMLSTM.Config()
        metric_reporter: LanguageModelMetricReporter.Config = (
            LanguageModelMetricReporter.Config()
        )


class PairwiseClassificationTask(NewTask):
    class Config(NewTask.Config):
        model: BasePairwiseModel.Config = PairwiseModel.Config()
        metric_reporter: ClassificationMetricReporter.Config = (
            ClassificationMetricReporter.Config(text_column_names=["text1", "text2"])
        )


class SeqNNTask_Deprecated(Task_Deprecated):
    class Config(Task_Deprecated.Config):
        model: SeqNNModel_Deprecated.Config = SeqNNModel_Deprecated.Config()
        trainer: Trainer.Config = Trainer.Config()
        labels: DocLabelConfig = DocLabelConfig()
        data_handler: SeqModelDataHandler.Config = SeqModelDataHandler.Config()
        metric_reporter: ClassificationMetricReporter.Config = (
            ClassificationMetricReporter.Config()
        )
        exporter: Optional[DenseFeatureExporter.Config] = None


class SeqNNTask(NewTask):
    class Config(NewTask.Config):
        model: SeqNNModel.Config = SeqNNModel.Config()
        metric_reporter: ClassificationMetricReporter.Config = (
            ClassificationMetricReporter.Config(text_column_names=["text_seq"])
        )


class ContextualIntentSlotTask_Deprecated(Task_Deprecated):
    class Config(Task_Deprecated.Config):
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
        exporter: Optional[DenseFeatureExporter.Config] = None

    @classmethod
    def example_config(cls):
        return cls.Config(labels=[DocLabelConfig(), WordLabelConfig()])


class SemanticParsingTask_Deprecated(Task_Deprecated):
    class Config(Task_Deprecated.Config):
        model: RNNGParser_Deprecated.Config = RNNGParser_Deprecated.Config()
        trainer: HogwildTrainer_Deprecated.Config = HogwildTrainer_Deprecated.Config()
        data_handler: CompositionalDataHandler.Config = CompositionalDataHandler.Config()
        labels: Optional[WordLabelConfig] = None
        metric_reporter: CompositionalMetricReporter.Config = CompositionalMetricReporter.Config()


class SemanticParsingTask(NewTask):
    class Config(NewTask.Config):
        model: RNNGParser.Config = RNNGParser.Config()
        trainer: HogwildTrainer.Config = HogwildTrainer.Config()
        metric_reporter: CompositionalMetricReporter.Config = CompositionalMetricReporter.Config()

    def __init__(
        self,
        data: Data,
        model: RNNGParser,
        metric_reporter: CompositionalMetricReporter,
        trainer: HogwildTrainer,
    ):
        super().__init__(data, model, metric_reporter, trainer)
        assert (
            (data.batcher.train_batch_size == 1)
            and (data.batcher.eval_batch_size == 1)
            and (data.batcher.test_batch_size == 1)
        ), "RNNGParser only supports batch size = 1"
        assert (
            trainer.config.report_train_metrics is False
        ), "Disable report_train_metrics because trees are not necessarily valid during training"
