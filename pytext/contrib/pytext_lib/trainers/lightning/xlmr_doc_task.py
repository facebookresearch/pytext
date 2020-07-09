#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from pytext.contrib.pytext_lib.datasets import TsvDataset, fb_hive_dataset
from pytext.contrib.pytext_lib.datasets.base_dataset import (
    PoolingBatcher,
    roberta_collate_fn,
)
from pytext.contrib.pytext_lib.datasets.fb_hive_dataset import HiveDataset
from pytext.contrib.pytext_lib.models.roberta import build_model
from pytext.contrib.pytext_lib.transforms import (
    LabelTransform,
    Transform,
    TruncateTransform,
    VocabTransform,
    build_fairseq_vocab,
)
from pytext.contrib.pytext_lib.transforms.fb_transforms import SpmTokenizerTransform
from pytext.fb.optimizer import FairSeqAdam
from torch.utils.data import DataLoader


# @manual=//github/third-party/PyTorchLightning/pytorch-lightning:lib
from pytorch_lightning import LightningModule  # noqa isort:skip


class XlmrForDocClassificationTask(LightningModule):
    # TorchScript requires attributes defined in class level
    infer_transforms: Dict[str, List[Transform]]
    model: nn.Module

    def __init__(self, config, global_rank: int = 0, word_size: int = 1):
        super().__init__()
        self.config = config
        self.infer_transforms = {}
        self.model = None

        self.global_rank = global_rank
        self.word_size = word_size

    def prepare(self):
        (self.train_transforms, self.infer_transforms) = self._build_transforms(
            **self.config.transforms
        )
        (
            self._train_dataloader,
            self._val_dataloader,
            self._test_dataloader,
        ) = self._build_dataloaders(self.train_transforms, **self.config.data)
        self.model = build_model(**self.config.model)
        self.optimizer = FairSeqAdam(self.model.parameters(), **self.config.optimizer)

    def forward(self, texts: List[str]) -> List[torch.Tensor]:
        model_outputs: List[torch.Tensor] = []
        for text in texts:
            tranformed = text
            for transform in self.infer_transforms["text"]:
                tranformed = transform(tranformed)
            model_outputs.append(self.model(tranformed))
        return model_outputs

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        targets = batch["label_ids"]
        loss = self.model.get_loss(logits, targets)
        batch_size = len(targets)
        loss = loss / batch_size
        return {"loss": loss, "log": {"train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)["loss"]
        return {"val_loss": loss, "log": {"val_loss": loss}}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def export(self):
        # export to TorchScript
        # TODO: @stevenliu make self.forward jitable
        return torch.jit.script(self)

    def _build_transforms(
        self, label_vocab: List[str], vocab_path: str, max_seq_len: int = 256
    ):
        # Custom batching and sampling will be setup here
        vocab = build_fairseq_vocab(vocab_path)
        label_transform = LabelTransform(label_vocab)
        train_transforms = {
            "text": [
                SpmTokenizerTransform(),
                VocabTransform(vocab),
                TruncateTransform(
                    vocab.get_bos_index(),
                    vocab.get_eos_index(),
                    max_seq_len=max_seq_len,
                ),
            ],
            "label": [label_transform],
        }
        infer_transforms = {"text": train_transforms["text"]}
        return train_transforms, infer_transforms

    def _build_dataloaders(
        self,
        train_transforms,
        train_path: str,
        val_path: Optional[str],
        test_path: Optional[str],
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
                    transforms_dict=train_transforms,
                    batcher=PoolingBatcher(batch_size=batch_size),
                    collate_fn=roberta_collate_fn,
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
        self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = True
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
            world_size=world_size
        )
        super().init_ddp_connection(global_rank, world_size, is_slurm_managing_tasks)
