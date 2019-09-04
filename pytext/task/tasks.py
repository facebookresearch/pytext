#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Union

from pytext.common.constants import DatasetFieldName, Stage
from pytext.config import doc_classification as DocClassification
from pytext.config.field_config import DocLabelConfig, TargetConfigBase, WordLabelConfig
from pytext.config.pytext_config import PlaceHolder
from pytext.data import (
    CompositionalDataHandler,
    DocClassificationDataHandler,
    JointModelDataHandler,
    SeqModelDataHandler,
)
from pytext.data.bert_tensorizer import BERTTensorizer
from pytext.data.data import Data
from pytext.data.packed_lm_data import PackedLMData
from pytext.data.tensorizers import Tensorizer
from pytext.exporters import DenseFeatureExporter
from pytext.metric_reporters import (
    ClassificationMetricReporter,
    CompositionalMetricReporter,
    IntentSlotMetricReporter,
    LanguageModelMetricReporter,
    MultiLabelClassificationMetricReporter,
    PairwiseRankingMetricReporter,
    PureLossMetricReporter,
    RegressionMetricReporter,
    SequenceTaggingMetricReporter,
    SquadMetricReporter,
    WordTaggingMetricReporter,
)
from pytext.metric_reporters.language_model_metric_reporter import (
    MaskedLMMetricReporter,
)
from pytext.models.bert_classification_models import NewBertModel
from pytext.models.bert_regression_model import NewBertRegressionModel
from pytext.models.doc_model import DocModel, DocModel_Deprecated, DocRegressionModel
from pytext.models.ensembles import BaggingDocEnsembleModel, EnsembleModel
from pytext.models.joint_model import IntentSlotModel
from pytext.models.language_models.lmlstm import LMLSTM
from pytext.models.masked_lm import MaskedLanguageModel
from pytext.models.model import BaseModel
from pytext.models.pair_classification_model import BasePairwiseModel, PairwiseModel
from pytext.models.qna.bert_squad_qa import BertSquadQAModel
from pytext.models.qna.dr_qa import DrQAModel
from pytext.models.query_document_pairwise_ranking_model import (
    QueryDocPairwiseRankingModel,
)
from pytext.models.representations.sparse_transformer_sentence_encoder import (  # noqa f401
    SparseTransformerSentenceEncoder,
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
    HogwildTrainer,
    HogwildTrainer_Deprecated,
    Trainer,
)


class QueryDocumentPairwiseRankingTask(NewTask):
    class Config(NewTask.Config):
        model: QueryDocPairwiseRankingModel.Config = (
            QueryDocPairwiseRankingModel.Config()
        )
        metric_reporter: PairwiseRankingMetricReporter.Config = (
            PairwiseRankingMetricReporter.Config()
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
        metric_reporter: Union[
            ClassificationMetricReporter.Config, PureLossMetricReporter.Config
        ] = (ClassificationMetricReporter.Config())
        #   for multi-label classification task,
        #   choose MultiLabelClassificationMetricReporter

    @classmethod
    def format_prediction(cls, predictions, scores, context, target_names):
        for prediction, score in zip(predictions, scores):
            score_with_name = {n: s for n, s in zip(target_names, score.tolist())}
            yield {
                "prediction": target_names[prediction.data],
                "score": score_with_name,
            }


class DocumentRegressionTask(NewTask):
    class Config(NewTask.Config):
        model: DocRegressionModel.Config = DocRegressionModel.Config()
        metric_reporter: RegressionMetricReporter.Config = (
            RegressionMetricReporter.Config()
        )


class NewBertClassificationTask(DocumentClassificationTask):
    class Config(DocumentClassificationTask.Config):
        model: NewBertModel.Config = NewBertModel.Config()


class NewBertPairClassificationTask(DocumentClassificationTask):
    class Config(DocumentClassificationTask.Config):
        model: NewBertModel.Config = NewBertModel.Config(
            inputs=NewBertModel.Config.BertModelInput(
                tokens=BERTTensorizer.Config(
                    columns=["text1", "text2"], max_seq_len=128
                )
            )
        )
        metric_reporter: ClassificationMetricReporter.Config = (
            ClassificationMetricReporter.Config(text_column_names=["text1", "text2"])
        )


class BertPairRegressionTask(DocumentRegressionTask):
    class Config(DocumentRegressionTask.Config):
        model: NewBertRegressionModel.Config = NewBertRegressionModel.Config()


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


class IntentSlotTask(NewTask):
    class Config(NewTask.Config):
        model: IntentSlotModel.Config = IntentSlotModel.Config()
        metric_reporter: IntentSlotMetricReporter.Config = (
            IntentSlotMetricReporter.Config()
        )


class LMTask(NewTask):
    class Config(NewTask.Config):
        model: LMLSTM.Config = LMLSTM.Config()
        metric_reporter: LanguageModelMetricReporter.Config = (
            LanguageModelMetricReporter.Config()
        )


class MaskedLMTask(NewTask):
    class Config(NewTask.Config):
        data: Data.Config = PackedLMData.Config()
        model: MaskedLanguageModel.Config = MaskedLanguageModel.Config()
        metric_reporter: MaskedLMMetricReporter.Config = (
            MaskedLMMetricReporter.Config()
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


class SquadQATask(NewTask):
    class Config(NewTask.Config):
        model: Union[BertSquadQAModel.Config, DrQAModel.Config] = DrQAModel.Config()
        metric_reporter: SquadMetricReporter.Config = SquadMetricReporter.Config()


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
