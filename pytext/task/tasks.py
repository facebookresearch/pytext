#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, Union

from pytext.common.constants import Stage
from pytext.data.bert_tensorizer import BERTTensorizer
from pytext.data.data import Data
from pytext.data.packed_lm_data import PackedLMData
from pytext.data.tensorizers import Tensorizer
from pytext.metric_reporters import (
    ClassificationMetricReporter,
    CompositionalMetricReporter,
    DenseRetrievalMetricReporter,
    IntentSlotMetricReporter,
    LanguageModelMetricReporter,
    NERMetricReporter,
    PairwiseRankingMetricReporter,
    PureLossMetricReporter,
    RegressionMetricReporter,
    SequenceTaggingMetricReporter,
    SquadMetricReporter,
)
from pytext.metric_reporters.channel import ConsoleChannel
from pytext.metric_reporters.language_model_metric_reporter import (
    MaskedLMMetricReporter,
)
from pytext.metric_reporters.seq2seq_compositional import (
    Seq2SeqCompositionalMetricReporter,
)
from pytext.models.bert_classification_models import NewBertModel
from pytext.models.bert_regression_model import NewBertRegressionModel
from pytext.models.doc_model import DocModel, DocRegressionModel
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
from pytext.models.roberta import RoBERTaWordTaggingModel
from pytext.models.semantic_parsers.rnng.rnng_parser import RNNGParser
from pytext.models.seq_models.contextual_intent_slot import (  # noqa
    ContextualIntentSlotModel,
)
from pytext.models.seq_models.seq2seq_model import Seq2SeqModel
from pytext.models.seq_models.seqnn import SeqNNModel
from pytext.models.word_model import WordTaggingModel
from pytext.task.new_task import NewTask
from pytext.trainers import EnsembleTrainer, HogwildTrainer


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
        return self.trainer.real_trainers[model_id].train(
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
        model: BaseModel.Config = DocRegressionModel.Config()
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


class WordTaggingTask(NewTask):
    class Config(NewTask.Config):
        model: WordTaggingModel.Config = WordTaggingModel.Config()
        metric_reporter: SequenceTaggingMetricReporter.Config = (
            SequenceTaggingMetricReporter.Config()
        )

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


class PairwiseClassificationForDenseRetrievalTask(PairwiseClassificationTask):
    """This task is to implement DPR training in PyText.
    Code pointer: https://github.com/fairinternal/DPR/tree/master/dpr
    """

    class Config(PairwiseClassificationTask.Config):
        metric_reporter: DenseRetrievalMetricReporter.Config = (
            DenseRetrievalMetricReporter.Config()
        )

    @classmethod
    def create_metric_reporter(cls, config: Config, *args, **kwargs):
        config.metric_reporter.task_batch_size = config.data.batcher.train_batch_size
        config.metric_reporter.num_negative_ctxs = config.data.source.num_negative_ctxs
        return super().create_metric_reporter(config, *args, **kwargs)


class RoBERTaNERTask(NewTask):
    class Config(NewTask.Config):
        model: RoBERTaWordTaggingModel.Config = RoBERTaWordTaggingModel.Config()
        metric_reporter: NERMetricReporter.Config = NERMetricReporter.Config()

    @classmethod
    def create_metric_reporter(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        return NERMetricReporter(
            channels=[ConsoleChannel()],
            label_names=list(tensorizers["tokens"].labels_vocab._vocab),
            pad_idx=tensorizers["tokens"].labels_pad_idx,
        )


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


class SemanticParsingTask(NewTask):
    class Config(NewTask.Config):
        model: RNNGParser.Config = RNNGParser.Config()
        trainer: HogwildTrainer.Config = HogwildTrainer.Config()
        metric_reporter: CompositionalMetricReporter.Config = (
            CompositionalMetricReporter.Config()
        )

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
        assert trainer.config.report_train_metrics is False, (
            "Disable report_train_metrics because trees are not necessarily "
            "valid during training"
        )


class SequenceLabelingTask(NewTask):
    class Config(NewTask.Config):
        model: Seq2SeqModel.Config = Seq2SeqModel.Config()
        metric_reporter: Seq2SeqCompositionalMetricReporter.Config = (
            Seq2SeqCompositionalMetricReporter.Config()
        )

    def torchscript_export(self, model, export_path, quantize=False):
        model.cpu()
        # Trace needs eval mode, to disable dropout etc
        model.eval()
        if hasattr(model, "torchscriptify"):
            jit_module = model.torchscriptify()
            jit_module.save(export_path)
