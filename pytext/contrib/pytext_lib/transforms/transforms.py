#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch.nn as nn
from pytext.contrib.pytext_lib.resources import url
from pytext.data.bert_tensorizer import build_fairseq_vocab
from pytext.data.utils import BOS, EOS, MASK, PAD, UNK
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils.file_io import PathManager
from torchtext.data.functional import load_sp_model


class IdentityTransform(nn.Module):
    def forward(self, **kwargs) -> Any:
        return kwargs


class RowsToColumnarTransform(nn.Module):
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


class WhitespaceTokenizerTransform(nn.Module):
    def forward(self, text: List[str]) -> List[List[str]]:
        return [t.split() for t in text]


SPECIAL_TOKEN_REPLACEMENT = {
    "[UNK]": UNK,
    "[PAD]": PAD,
    "[CLS]": BOS,
    "[MASK]": MASK,
    "[SEP]": EOS,
}


class VocabTransform(nn.Module):
    def __init__(
        self,
        vocab_path: Optional[str] = None,
        vocab_list: Optional[List[str]] = None,
        special_token_replacements=SPECIAL_TOKEN_REPLACEMENT,
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
                vocab = build_fairseq_vocab(
                    f, special_token_replacements=special_token_replacements
                )
                self.vocab = ScriptVocabulary(
                    list(vocab),
                    pad_idx=vocab.get_pad_index(-1),
                    bos_idx=vocab.get_bos_index(-1),
                    eos_idx=vocab.get_eos_index(-1),
                    unk_idx=vocab.get_unk_index(-1),
                    unk_token=vocab.unk_token,
                )

    def forward(self, tokens: List[List[str]]) -> List[List[int]]:
        return self.vocab.lookup_indices_2d(tokens)


class LabelTransform(nn.Module):
    def __init__(self, label_names: List[str]):
        super().__init__()

        self.label_vocab = ScriptVocabulary(label_names)

    def forward(self, labels: List[str]) -> List[int]:
        return self.label_vocab.lookup_indices_1d(labels)


class TruncateTransform(nn.Module):
    def __init__(self, max_seq_len: int):
        super().__init__()
        assert max_seq_len > 0
        self.max_seq_len: int = max_seq_len

    def forward(self, token_ids: List[List[int]]) -> List[List[int]]:
        return [token_id[: self.max_seq_len] for token_id in token_ids]


class SpmTokenizerTransform(nn.Module):
    """
    Uses TorchText SPM tokenizer
    """

    def __init__(self, sp_model_path: Optional[str] = None):
        super().__init__()

        # This default spm file path is a dummy link as we haven't published
        # the file yet. Please provide your own spm file path when using
        # this transform
        sp_model_path = sp_model_path or url.URL[url.SP_MODEL]
        local_path = PathManager.get_local_path(sp_model_path)
        self.sp_model = load_sp_model(local_path)

    def forward(self, texts: List[str]) -> List[List[str]]:
        tokens: List[List[str]] = []
        for text in texts:
            tokens.append(self.sp_model.EncodeAsPieces(text))
        return tokens
