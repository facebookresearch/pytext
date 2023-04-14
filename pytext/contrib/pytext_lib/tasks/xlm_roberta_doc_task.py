#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List

from pytext.contrib.pytext_lib import models
from pytext.contrib.pytext_lib.datasets import TsvDataset
from pytext.contrib.pytext_lib.trainers import SimpleTrainer
from pytext.contrib.pytext_lib.transforms import (
    CapTransform,
    LabelTransform,
    RobertaInputTransform,
    SpmTokenizerTransform,
    VocabTransform,
    build_vocab,
)
from pytext.data.utils import Vocabulary
from pytext.fb.optimizer import FairSeqAdam
from torch.utils.data import DataLoader


class XLMRobertaForDocClassificationTask(object):
    def __init__(
        self,
        # Data config
        train_data_path: str,
        valid_data_path: str,
        test_data_path: str,
        label_vocab: List[str],
        column_names: List[str] = None,
        text_column: str = "text",
        label_column: str = "label",
        batch_size: int = 1,
        vocab_path: str = "manifold://nlp_technologies/tree/xlm/models/xlm_r/vocab",
        vocab_size: int = 250002,
        # Model config
        model_name: str = "xlm_roberta_base",
        # Trainer config
        epoch: int = 2,
        # TODO: Meric config
    ):
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.test_data_path = test_data_path

        self.column_names = column_names
        self.text_column = text_column
        self.label_column = label_column
        self.label_column = label_column
        self.label_vocab = label_vocab
        self.batch_size = batch_size

        self.vocab_path = vocab_path
        self.vocab_size = vocab_size

        self.model_name = model_name

        self.epoch = epoch

    def prepare(self):
        self._build_transforms()
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = self._build_data(
            data_paths=[self.train_data_path, self.valid_data_path, self.test_data_path]
        )
        self.model = self._build_model(self.model_name)
        self.optimizer = FairSeqAdam(
            self.model.parameters(),
            lr=0.00001,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0,
            amsgrad=False,
        )
        self.trainer = SimpleTrainer()

    def train(self):
        self.trainer.train(
            model=self.model,
            optimizer=self.optimizer,
            dataloader=self.train_dataloader,
            epoch=self.epoch,
        )

    def test(self):
        pass

    def export(self):
        # export self.input_transform and self.model to TorchScript
        pass

    def predict(self, batch_inputs):
        self.model.eval()
        texts = self.input_transform.extract_inputs(batch_inputs)
        model_inputs = self.input_transform(texts)
        logits = self.model(model_inputs)
        return self.model.get_pred(logits)

    def _build_transforms(self):
        tokenizer_transform = SpmTokenizerTransform()
        vocab = build_vocab(self.vocab_path)
        vocab_transform = VocabTransform(vocab)
        cap_transform = CapTransform(
            vocab.get_bos_index(), vocab.get_eos_index(), max_seq_len=256
        )
        transforms = [tokenizer_transform, vocab_transform, cap_transform]
        self.input_transform = RobertaInputTransform(transforms, 1, self.text_column)

        label_vocab = Vocabulary(self.label_vocab)
        self.label_transform = LabelTransform(
            label_vocab, field_name=self.label_column, pad_idx=-1
        )

    def _build_data(self, data_paths):
        # Custom batching and sampling will be setup here
        dataloaders = []
        for data_path in data_paths:
            dataset = TsvDataset(
                file_path=data_path,
                batch_size=self.batch_size,
                field_names=self.column_names,
                transform=self.input_transform,
                label_transform=self.label_transform,
            )
            # must set batch_size=None to skip batching in DataLoader
            dataloaders.append(DataLoader(dataset, batch_size=None))
        return dataloaders

    def _build_model(self, model_name):
        if model_name == "xlm_roberta_base":
            return models.xlm_roberta_base_binary_doc_classifier(pretrained=True)
        elif model_name == "xlm_roberta_dummy":
            return models.xlm_roberta_dummy_binary_doc_classifier(pretrained=False)
        return models.xlm_roberta_base_binary_doc_classifier(pretrained=True)
