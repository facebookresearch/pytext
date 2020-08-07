#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from pytext.contrib.pytext_lib.datasets import TsvDataset, fb_hive_dataset
from pytext.contrib.pytext_lib.datasets.base_dataset import (
    PoolingBatcher,
    intent_slot_collate_fn,
)
from pytext.contrib.pytext_lib.datasets.fb_hive_dataset import HiveDataset
from pytext.contrib.pytext_lib.models.intent_slot_model import (
    build_slot_labelling_model,
)
from pytext.contrib.pytext_lib.transforms.transforms import (
    LabelTransform,
    SlotLabelTransform,
    TokenizerTransform,
    Transform,
    TruncateTransform,
    VocabTransform,
    WhiteSpaceTokenizer,
    build_fairseq_vocab,
)
from pytext.fb.optimizer import FairSeqAdam
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import F1, Accuracy, AveragePrecision
from torch import Tensor
from torch.utils.data import DataLoader


class SlotLabellingTask(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.slot_model: nn.Module = None
        self.infer_transforms: Dict[str, List[Transform]] = {}
        self.train_transforms: Dict[str, List[Transform]] = {}
        self.accuracy_list: List[Tensor] = []
        self.precision_list: List[Tensor] = []
        self.f1_list: List[Tensor] = []

    def prepare(self):
        (self.train_transforms, self.infer_transforms) = self._build_transforms(
            **self.config.transforms
        )
        (
            self._train_dataloader,
            self._val_dataloader,
            self._test_dataloader,
        ) = self._build_dataloaders(**self.config.data)
        num_slots = len(self.train_transforms["utterance&slots"][0].vocab.idx.keys())
        num_intents = len(self.train_transforms["intent"][0].vocab.idx.keys())
        self.slot_model = build_slot_labelling_model(
            **self.config.model,
            num_slots=num_slots,
            num_intents=num_intents,
            vocab=self.train_transforms["utterance"][1].vocab,
        )
        self.accuracy_metric = Accuracy(num_classes=num_slots)
        self.precision_metric = AveragePrecision()
        self.f1_metric = F1(num_classes=num_slots)
        self.optimizer = FairSeqAdam(
            self.slot_model.parameters(), **self.config.optimizer
        )

    def forward(self, texts: List[str]) -> List[torch.Tensor]:
        model_outputs: List[torch.Tensor] = []
        for text in texts:
            transformed_text = text
            for transform in self.infer_transforms["utterance"]:
                transformed_text = transform(transformed_text)
            slot_prediction = self.slot_model(transformed_text)
            model_outputs.append([slot_prediction])
        return model_outputs

    def training_step(self, batch, batch_idx):
        logits = self.slot_model(batch, batch["label_ids"])
        prediction = self.slot_model.get_pred(logits)[0]
        targets = batch["slot_label_ids"].reshape_as(prediction)
        self.set_metrics(prediction, targets)
        loss = self.slot_model.get_loss(logits, targets)
        batch_size = len(targets)
        loss = loss / batch_size
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        train_step = self.training_step(batch, batch_idx)
        loss = train_step["loss"]
        return {"val_loss": loss, "log": {"val_loss": loss}}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return self.optimizer

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def export(self):
        # export to TorchScript
        # TODO: make self.forward jitable
        return torch.jit.script(self)

    def get_metrics(self, last_n_batches: int = 1):
        start_index = len(self.accuracy_list) - (last_n_batches + 1)
        if start_index < 0:
            start_index = 0
        ret_accuracy = self.accuracy_list[start_index:]
        ret_precision = self.precision_list[start_index:]
        ret_f1 = self.f1_list[start_index:]
        print("last ", last_n_batches, " metrics")
        print("accuracy ", ret_accuracy)
        print("precision ", ret_precision)
        print("f1 ", ret_f1)
        return ret_accuracy, ret_precision, ret_f1

    def set_metrics(self, preds, targets):
        accuracy = self.accuracy_metric(preds, targets)
        self.accuracy_list.append(accuracy)
        print("batch accuracy ", accuracy)
        precision = self.precision_metric(preds, targets)
        self.precision_list.append(precision)
        f1 = self.f1_metric(preds, targets)
        self.f1_list.append(f1)

    def _build_transforms(
        self,
        doc_label_vocab: List[str],
        slot_label_vocab: List[str],
        vocab_path: str,
        max_seq_len: int = 256,
    ):
        # Custom batching and sampling will be setup here
        vocab = build_fairseq_vocab(vocab_path)
        doc_label_transform = LabelTransform(doc_label_vocab)
        slot_label_transform = SlotLabelTransform(slot_label_vocab)
        train_transforms = {
            "utterance": [
                TokenizerTransform(WhiteSpaceTokenizer()),
                VocabTransform(vocab),
                TruncateTransform(
                    vocab.get_bos_index(),
                    vocab.get_eos_index(),
                    max_seq_len=max_seq_len,
                ),
            ],
            "intent": [doc_label_transform],
            "utterance&slots": [slot_label_transform],  # needs slot labels and text
        }
        infer_transforms = {"utterance": train_transforms["utterance"]}
        return train_transforms, infer_transforms

    def _build_dataloaders(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        field_names: List[str],
        batch_size: int = 8,
        is_shuffle: bool = True,
        is_cycle: bool = False,
        length: Optional[int] = None,
        # args for distributed training
        global_rank: int = 0,
        world_size: int = 1,
    ):
        datasets = []
        for path in (train_path, val_path, test_path):
            if not path:
                continue
            dataset_class = (
                HiveDataset
                if path.startswith(fb_hive_dataset.HIVE_PREFIX)
                else TsvDataset
            )
            datasets.append(
                dataset_class(
                    path=path,
                    field_names=field_names,
                    is_shuffle=is_shuffle,
                    transforms_dict=self.train_transforms,
                    batcher=PoolingBatcher(batch_size=batch_size),
                    collate_fn=intent_slot_collate_fn,
                    is_cycle=is_cycle,
                    length=length,
                    rank=global_rank,
                    num_workers=world_size,
                )
            )
        _train_dataloader = DataLoader(datasets[0], batch_size=None)
        _val_dataloader = DataLoader(datasets[1], batch_size=None) if val_path else None
        _test_dataloader = (
            DataLoader(datasets[2], batch_size=None) if test_path else None
        )
        return _train_dataloader, _val_dataloader, _test_dataloader

    def init_ddp_connection(
        self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = False
    ) -> None:
        # overwrite dataloaders with data sharding for distributed training
        self.global_rank = global_rank
        self.word_size = world_size
        (
            self._train_dataloader,
            self._val_dataloader,
            self._test_dataloaders,
        ) = self._build_dataloaders(
            train_transforms=self.train_transforms,
            **self.config.data,
            global_rank=global_rank,
            world_size=world_size,
        )
        super().init_ddp_connection(global_rank, world_size, is_slurm_managing_tasks)
