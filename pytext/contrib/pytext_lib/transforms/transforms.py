#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch.nn as nn
from pytext.data.bert_tensorizer import build_fairseq_vocab
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils.file_io import PathManager


class Transform(nn.Module):
    def forward(self, x: Any) -> Any:
        raise NotImplementedError()


class ScriptTransform(Transform):
    def forward(self, x):
        """If your transform is TorchScriptable, extend this"""
        raise NotImplementedError()


class IdentityTransform(Transform):
    def forward(self, x: Any) -> Any:
        return x


class RowsToColumnarTransform(Transform):
    """Adapter to process rows format with columnar transform

    Many datasets are stored in rows format. It's flexible to manipulate(e.g. batching).
    However, it's problematic to TorchScript as columns could have different format
    and TorchScript doesn't support Dict[str, Any].
    e.g. [{"text": "a b", "label": "c"}, {"text": "x y", "label": "z"}]

    Columnar format is less flexible but TorchScript friendly, and it's more memory
    efficent.
    e.g. (text=[["a b", "x y"]], label=["c", "z"])

    we use rows format during training and columnar during inference. The main logic
    is shared by this adapter

    Pitfall: it adds additional pass on the data during training
    """

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, rows: List[Dict[str, Any]]) -> Any:
        columnar = defaultdict(list)
        for row in rows:
            for column, value in row.items():
                columnar[column].append(value)
        return self.transform(**columnar)


class WhitespaceTokenizerTransform(ScriptTransform):
    def forward(self, text: List[str]) -> List[List[str]]:
        return [t.split() for t in text]


class VocabTransform(ScriptTransform):
    def __init__(
        self, vocab_path: Optional[str] = None, vocab_list: Optional[List[str]] = None
    ):
        super().__init__()
        assert vocab_path or vocab_list, "vocab_path or vocab_list is required"
        assert not (
            vocab_path and vocab_list
        ), "vocab_path and vocab_list are mutual exclusive"

        if vocab_list:
            self.vocab = ScriptVocabulary(vocab_list)
        else:
            with PathManager.open(vocab_path) as f:
                vocab = build_fairseq_vocab(f)
                self.vocab = ScriptVocabulary(
                    list(vocab),
                    pad_idx=vocab.get_pad_index(-1),
                    bos_idx=vocab.get_bos_index(-1),
                    eos_idx=vocab.get_eos_index(-1),
                    unk_idx=vocab.get_unk_index(-1),
                )

    def forward(self, tokens: List[List[str]]) -> List[List[int]]:
        return self.vocab.lookup_indices_2d(tokens)


class LabelTransform(ScriptTransform):
    def __init__(self, label_names: List[str]):
        super().__init__()

        self.label_vocab = ScriptVocabulary(label_names)

    def forward(self, labels: List[str]) -> List[int]:
        return self.label_vocab.lookup_indices_1d(labels)
