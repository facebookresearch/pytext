#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from fairseq.data.encoders.gpt2_bpe import get_encoder
from pytext.data.bert_tensorizer import build_fairseq_vocab
from pytext.data.tokenizers import GPT2BPETokenizer
from pytext.data.utils import BOS, EOS, MASK, PAD, UNK, Vocabulary, pad_and_tensorize
from pytext.fb.torchscript.tokenizer import ScriptSentencePieceTokenizer
from pytext.torchscript.utils import ScriptBatchInput, pad_2d, pad_2d_mask
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils.file_io import PathManager


class Tokens(NamedTuple):
    values: List[str]
    start_idxs: List[int]
    end_idxs: List[int]


class TokenIds(NamedTuple):
    values: List[int]
    start_idxs: List[int]
    end_idxs: List[int]


def build_vocab(vocab_file):
    with PathManager.open(vocab_file) as f:
        vocab = build_fairseq_vocab(
            vocab_file=f,
            special_token_replacements={
                "<pad>": PAD,
                "<s>": BOS,
                "</s>": EOS,
                "<unk>": UNK,
                "<mask>": MASK,
            },
        )
    return vocab


class TokenizerTransform(nn.Module):
    def __init__(self, tokenizer: nn.Module):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, text_batch: List[str]) -> List[Tokens]:
        tokens_batch: List[Tokens] = []
        for text in text_batch:
            tokens: List[str] = []
            start_idxs: List[int] = []
            end_idxs: List[int] = []
            for token_with_index in self.tokenizer.tokenize(text):
                tokens.append(token_with_index[0])
                start_idxs.append(token_with_index[1])
                end_idxs.append(token_with_index[2])
            tokens_batch.append(Tokens(tokens, start_idxs, end_idxs))
        return tokens_batch


class SpaceTokenizer(nn.Module):
    def forward(self, text: str) -> List[Tuple[str, int, int]]:
        return self.tokenize(text)

    def tokenize(self, text: str) -> List[Tuple[str, int, int]]:
        tokens: List[Tuple[str, int, int]] = []
        words = text.split()
        cur_offset = 0
        for word in words:
            word_offset = text.index(word, cur_offset)
            cur_offset = word_offset + len(word)
            tokens.append((word, word_offset, word_offset + len(word)))
        return tokens


class SpmTokenizerTransform(TokenizerTransform):
    def __init__(self, sp_model_path=None):
        sp_model_path = (
            sp_model_path or "/mnt/vol/nlp_technologies/xlm/models/xlm_r/model"
        )
        torch.ops.load_library(
            "//caffe2/torch/fb/nlp/operators:sentencepiece_tokenizer"
        )
        with open(sp_model_path, "rb") as model_content:
            processor = torch.classes.fb.SentencePieceWithIndices(model_content.read())
        tokenizer = ScriptSentencePieceTokenizer(processor)

        super().__init__(tokenizer)


class Gpt2BpeTokenizerTransform(TokenizerTransform):
    def __init__(self, bpe_encoder_path=None, bpe_vocab_path=None):
        bpe_encoder_path = (
            bpe_encoder_path
            or "manifold://pytext_training/tree/static/vocabs/bpe/gpt2/encoder.json"
        )
        bpe_vocab_path = (
            bpe_vocab_path
            or "manifold://pytext_training/tree/static/vocabs/bpe/gpt2/vocab.bpe"
        )

        bpe_encoder_path = PathManager.get_local_path(bpe_encoder_path)
        bpe_vocab_path = PathManager.get_local_path(bpe_vocab_path)

        bpe = get_encoder(bpe_encoder_path, bpe_vocab_path)
        tokenizer = GPT2BPETokenizer(bpe)

        # Note: GPT2BPETokenizer is not a nn.Module
        super().__init__(tokenizer)


class VocabTransform(nn.Module):
    def __init__(self, vocab: Vocabulary):
        super().__init__()
        self.vocab = ScriptVocabulary(
            list(vocab),
            pad_idx=vocab.get_pad_index(),
            bos_idx=vocab.get_bos_index(-1),
            eos_idx=vocab.get_eos_index(-1),
            unk_idx=vocab.get_unk_index(),
        )

    def forward(self, tokens_batch: List[Tokens]) -> List[TokenIds]:
        token_ids_batch: List[TokenIds] = []
        for tokens in tokens_batch:
            # vocab lookup
            token_ids: List[int] = self.vocab.lookup_indices_1d(tokens.values)
            token_ids_batch.append(
                TokenIds(token_ids, tokens.start_idxs, tokens.end_idxs)
            )
        return token_ids_batch


class CapTransform(nn.Module):
    def __init__(self, bos_idx: int, eos_idx: int, max_seq_len: int):
        super().__init__()
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.max_seq_len = max_seq_len

    def forward(self, token_ids_batch: List[TokenIds]) -> List[TokenIds]:
        capped_batch: List[TokenIds] = []
        max_seq_len = (
            self.max_seq_len
            - (1 if self.bos_idx >= 0 else 0)
            - (1 if self.eos_idx >= 0 else 0)
        )

        for token_ids in token_ids_batch:
            ids = token_ids.values[0:max_seq_len]
            start_idxs = token_ids.start_idxs[0:max_seq_len]
            end_idxs = token_ids.end_idxs[0:max_seq_len]

            # add bos and eos index if needed
            if self.bos_idx >= 0:
                ids.insert(0, self.bos_idx)
                start_idxs = [-1] + start_idxs
                end_idxs = [-1] + end_idxs
            if self.eos_idx >= 0:
                ids.append(self.eos_idx)
                start_idxs.append(-1)
                end_idxs.append(-1)
            capped_batch.append(TokenIds(ids, start_idxs, end_idxs))
        return capped_batch


class RobertaTensorTransform(nn.Module):
    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx: int = pad_idx

    def forward(
        self,
        tokens_2d: List[List[int]],
        segment_labels_2d: List[List[int]],
        seq_lens_1d: List[int],
        positions_2d: List[List[int]],
    ) -> Dict[str, torch.Tensor]:
        tokens, pad_mask = pad_2d_mask(tokens_2d, pad_value=self.pad_idx)
        segment_labels = torch.tensor(
            pad_2d(segment_labels_2d, seq_lens=seq_lens_1d, pad_idx=0), dtype=torch.long
        )
        positions = torch.tensor(
            pad_2d(positions_2d, seq_lens=seq_lens_1d, pad_idx=0), dtype=torch.long
        )
        return {
            "tokens": tokens,
            "pad_mask": pad_mask,
            "segment_labels": segment_labels,
            "positions": positions,
        }


class RobertaInputTransform(nn.Module):
    def __init__(self, transforms: List[nn.Module], pad_idx: int, field_name: str):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.tensor_transform = RobertaTensorTransform(pad_idx)
        self.field_name = field_name

    def forward(self, text_batch: List[str]) -> Dict[str, torch.Tensor]:
        tokens_2d: List[List[int]] = []
        segment_labels_2d: List[List[int]] = []
        seq_lens_1d: List[int] = []
        positions_2d: List[List[int]] = []

        transformed_batch = text_batch
        for transform in self.transforms:
            transformed_batch = transform(transformed_batch)

        for transformed in transformed_batch:
            token_ids: List[int] = transformed.values
            tokens_2d.append(token_ids)
            segment_labels_2d.append([0] * len(token_ids))
            seq_lens_1d.append(len(token_ids))
            positions_2d.append(list(range(len(token_ids))))

        return self.tensor_transform(
            tokens_2d, segment_labels_2d, seq_lens_1d, positions_2d
        )

    @torch.jit.export
    def extract_inputs_jit(self, batch_input: ScriptBatchInput) -> List[str]:
        optional_texts: Optional[List[List[str]]] = batch_input.texts
        if optional_texts is not None:
            text_batch: List[str] = []
            for texts in optional_texts:
                text_batch.append(texts[0])
            return text_batch
        else:
            raise TypeError()

    def extract_inputs(self, batch: List[Dict[str, Any]]) -> List[str]:
        text_batch: List[int] = []
        for row in batch:
            text_batch.append(row[self.field_name])
        return text_batch


class LabelTransform(Callable):
    def __init__(self, vocab: Vocabulary, field_name: str, pad_idx: int):
        self.vocab = vocab
        self.field_name = field_name
        self.pad_idx = pad_idx

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = []
        for row in batch:
            labels.append(self.vocab.lookup_all(row[self.field_name]))
        return {"label": pad_and_tensorize(labels, self.pad_idx)}
