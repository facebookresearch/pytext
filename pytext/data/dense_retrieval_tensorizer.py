#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, List, Optional, Tuple

from pytext.common.constants import SpecialTokens
from pytext.config.component import ComponentType, create_component
from pytext.data.bert_tensorizer import BERTTensorizer, build_fairseq_vocab
from pytext.data.roberta_tensorizer import RoBERTaTensorizer
from pytext.data.tensorizers import LabelTensorizer
from pytext.data.tokenizers import Tokenizer
from pytext.data.utils import Vocabulary
from pytext.utils.file_io import PathManager


class BERTContextTensorizerForDenseRetrieval(BERTTensorizer):
    """Methods numberize() and tensorize() implement https://fburl.com/an4fv7m1."""

    def numberize(self, row: Dict) -> Tuple[Any, ...]:
        """
        This function contains logic for converting tokens into ids based on
        the specified vocab. It also outputs, for each instance, the vectors
        needed to run the actual model. It works off of one sample.
        """
        positive_ctx = row["positive_ctx"]
        positive_ctx_tokens = self.tokenizer.tokenize(positive_ctx)
        (
            positive_ctx_token_ids,
            positive_ctx_segment_labels,
            positive_ctx_seq_len,
            positive_ctx_positions,
        ) = self.tensorizer_script_impl.numberize([positive_ctx_tokens])

        negative_ctxs = row["negative_ctxs"]  # returns List[str]
        if negative_ctxs:
            negative_ctx_tokens = self.tokenizer.tokenize(
                negative_ctxs[: row["num_negative_ctx"]]
            )
            (
                negative_ctx_token_ids,
                negative_ctx_segment_labels,
                negative_ctx_seq_len,
                negative_ctx_positions,
            ) = self.tensorizer_script_impl.numberize(negative_ctx_tokens)
        else:
            negative_ctx_token_ids = []
            negative_ctx_segment_labels = []
            negative_ctx_seq_len = 0
            negative_ctx_positions = []

        return (
            positive_ctx_token_ids,
            positive_ctx_segment_labels,
            positive_ctx_seq_len,
            positive_ctx_positions,
            negative_ctx_token_ids,
            negative_ctx_segment_labels,
            negative_ctx_seq_len,
            negative_ctx_positions,
        )

    def tensorize(self, batch):
        """Works off of a batch that's numerized."""
        all_ctx_tokens_2d = []
        all_ctx_segment_labels_2d = []
        all_ctx_seq_lens_1d = []
        all_ctx_positions_2d = []
        for (
            positive_ctx_token_ids,
            positive_ctx_segment_labels,
            positive_ctx_seq_len,
            positive_ctx_positions,
            negative_ctx_token_ids,
            negative_ctx_segment_labels,
            negative_ctx_seq_len,
            negative_ctx_positions,
        ) in batch:
            # Make sure the positive and hard negative context for a given
            # question are one after another in the batch.
            all_ctx_tokens_2d.append(positive_ctx_token_ids)
            all_ctx_tokens_2d.append(negative_ctx_token_ids)
            all_ctx_segment_labels_2d.append(positive_ctx_segment_labels)
            all_ctx_segment_labels_2d.append(negative_ctx_segment_labels)
            all_ctx_seq_lens_1d.append(positive_ctx_seq_len)
            all_ctx_seq_lens_1d.append(negative_ctx_seq_len)
            all_ctx_positions_2d.append(positive_ctx_positions)
            all_ctx_positions_2d.append(negative_ctx_positions)

        return self.tensorizer_script_impl.tensorize_wrapper(
            all_ctx_tokens_2d,
            all_ctx_segment_labels_2d,
            all_ctx_seq_lens_1d,
            all_ctx_positions_2d,
        )


class RoBERTaContextTensorizerForDenseRetrieval(
    BERTContextTensorizerForDenseRetrieval, RoBERTaTensorizer
):
    class Config(RoBERTaTensorizer.Config):
        pass

    @classmethod
    def from_config(cls, config: Config):
        tokenizer = create_component(ComponentType.TOKENIZER, config.tokenizer)
        with PathManager.open(config.vocab_file) as file_path:
            vocab = build_fairseq_vocab(
                vocab_file=file_path,
                special_token_replacements={
                    "<pad>": SpecialTokens.PAD,
                    "<s>": SpecialTokens.BOS,
                    "</s>": SpecialTokens.EOS,
                    "<unk>": SpecialTokens.UNK,
                    "<mask>": SpecialTokens.MASK,
                },
            )
        return cls(
            columns=config.columns,
            vocab=vocab,
            tokenizer=tokenizer,
            max_seq_len=config.max_seq_len,
        )

    def __init__(
        self,
        columns: List[str] = Config.columns,
        vocab: Optional[Vocabulary] = None,
        tokenizer: Optional[Tokenizer] = None,
        max_seq_len: int = Config.max_seq_len,
    ):
        RoBERTaTensorizer.__init__(
            self,
            columns=columns,
            vocab=vocab,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )


class PositiveLabelTensorizerForDenseRetrieval(LabelTensorizer):
    def numberize(self, row: Dict):
        return row["num_negative_ctx"]

    def tensorize(self, batch):
        new_batch = []
        for i in range(len(batch)):
            # batch[i - 1] = No. of -ve ctxs in previous example; +1 for +ve ctx
            pos_ctx_idx = i if i == 0 else new_batch[-1] + batch[i - 1] + 1
            new_batch.append(pos_ctx_idx)
        return super().tensorize(new_batch)
