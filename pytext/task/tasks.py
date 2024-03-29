#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, Union

import torch
from pytext.common.constants import Stage
from pytext.config import ExportConfig
from pytext.config.component import create_trainer
from pytext.data.bert_tensorizer import BERTTensorizer
from pytext.data.data import Data
from pytext.data.decoupled_data import DecoupledSeq2SeqData
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
    TopKClassificationMetricReporter,
)
from pytext.metric_reporters.channel import ConsoleChannel
from pytext.metric_reporters.language_model_metric_reporter import (
    MaskedLMMetricReporter,
)
from pytext.metric_reporters.mask_compositional import (
    MaskedSeq2SeqCompositionalMetricReporter,
)
from pytext.metric_reporters.seq2seq_compositional import (
    Seq2SeqCompositionalMetricReporter,
)
from pytext.models.bert_classification_models import NewBertModel
from pytext.models.bert_regression_model import (
    BertPairwiseRegressionModel,
    NewBertRegressionModel,
)
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
from pytext.models.seq_models.nar_seq2seq_model import (
    NARDecoupledSeq2SeqModel,
    NARSeq2SeqModel,
)
from pytext.models.seq_models.seq2seq_model import Seq2SeqModel
from pytext.models.seq_models.seqnn import SeqNNModel
from pytext.models.word_model import WordTaggingModel
from pytext.task.new_task import NewTask
from pytext.trainers import EnsembleTrainer, HogwildTrainer, TaskTrainer
from pytext.utils import cuda
from pytext.utils.file_io import PathManager
from torch import jit


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
            ClassificationMetricReporter.Config,
            PureLossMetricReporter.Config,
            TopKClassificationMetricReporter.Config,
        ] = ClassificationMetricReporter.Config()
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
        metric_reporter: MaskedLMMetricReporter.Config = MaskedLMMetricReporter.Config()


class PairwiseClassificationTask(NewTask):
    class Config(NewTask.Config):
        model: BasePairwiseModel.Config = PairwiseModel.Config()
        metric_reporter: ClassificationMetricReporter.Config = (
            ClassificationMetricReporter.Config(text_column_names=["text1", "text2"])
        )
        trace_both_encoders: bool = True

    @classmethod
    def from_config(
        cls,
        config: Config,
        unused_metadata=None,
        model_state=None,
        tensorizers=None,
        rank=0,
        world_size=1,
    ):
        tensorizers, data = cls._init_tensorizers(config, tensorizers, rank, world_size)
        model = cls._init_model(config.model, tensorizers, model_state)
        metric_reporter = cls.create_metric_reporter(config, tensorizers)
        trainer = create_trainer(config.trainer, model)
        return cls(data, model, metric_reporter, trainer, config.trace_both_encoders)

    def __init__(
        self,
        data: Data,
        model: BaseModel,
        metric_reporter: ClassificationMetricReporter,
        trainer: TaskTrainer,
        trace_both_encoders: bool = True,
    ):
        super().__init__(data, model, metric_reporter, trainer)
        self.trace_both_encoders = trace_both_encoders

    def torchscript_export(self, model, export_path=None, export_config=None):  # noqa
        # unpack export config
        # unpack export config
        if export_config is None:
            export_config = ExportConfig()

        quantize = export_config.torchscript_quantize
        accelerate = export_config.accelerate
        seq_padding_control = export_config.seq_padding_control
        batch_padding_control = export_config.batch_padding_control

        if (accelerate is not None) and (accelerate != []):
            raise RuntimeError(
                "old-style task.py does not support export for NNPI accelerators"
            )

        cuda.CUDA_ENABLED = False
        model.cpu()
        optimizer = self.trainer.optimizer
        optimizer.pre_export(model)

        model.eval()
        model.prepare_for_onnx_export_()

        unused_raw_batch, batch = next(
            iter(self.data.batches(Stage.TRAIN, load_early=True))
        )
        inputs = model.onnx_trace_input(batch)
        model(*inputs)
        if quantize:
            model.quantize()
        if self.trace_both_encoders:
            trace = jit.trace(model, inputs)
        else:
            trace = jit.trace(model.encoder1, (inputs[0],))
        if hasattr(model, "torchscriptify"):
            trace = model.torchscriptify(
                self.data.tensorizers, trace, self.trace_both_encoders
            )
        if seq_padding_control is not None:
            if hasattr(trace, "set_padding_control"):
                trace.set_padding_control("sequence_length", seq_padding_control)
            else:
                print(
                    "Padding_control not supported by model. Ignoring padding_control"
                )
        if batch_padding_control is not None:
            if hasattr(trace, "set_padding_control"):
                trace.set_padding_control("batch_length", batch_padding_control)
            else:
                print(
                    "Padding_control not supported by model. Ignoring padding_control"
                )
        trace.apply(lambda s: s._pack() if s._c._has_method("_pack") else None)
        if export_path is not None:
            print(f"Saving torchscript model to: {export_path}")
            with PathManager.open(export_path, "wb") as f:
                torch.jit.save(trace, f)
        return trace


class PairwiseRegressionTask(PairwiseClassificationTask):
    class Config(PairwiseClassificationTask.Config):
        model: BasePairwiseModel.Config = BertPairwiseRegressionModel.Config()
        metric_reporter: RegressionMetricReporter.Config = (
            RegressionMetricReporter.Config()
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

    def torchscript_export(self, model, export_path=None, export_config=None):
        model.cpu()
        # Trace needs eval mode, to disable dropout etc
        model.eval()
        if hasattr(model, "torchscriptify"):
            jit_module = model.torchscriptify()
            with PathManager.open(export_path, "wb") as f:
                torch.jit.save(jit_module, f)


class NARSeq2SeqTask(NewTask):
    class Config(NewTask.Config):
        model: NARSeq2SeqModel.Config = NARSeq2SeqModel.Config()
        metric_reporter: MaskedSeq2SeqCompositionalMetricReporter.Config = (
            MaskedSeq2SeqCompositionalMetricReporter.Config()
        )


class DecoupledNARSeq2SeqTask(NewTask):
    class Config(NewTask.Config):
        data: DecoupledSeq2SeqData.Config = DecoupledSeq2SeqData.Config()
        model: NARDecoupledSeq2SeqModel.Config = NARDecoupledSeq2SeqModel.Config()
        metric_reporter: MaskedSeq2SeqCompositionalMetricReporter.Config = (
            MaskedSeq2SeqCompositionalMetricReporter.Config()
        )
