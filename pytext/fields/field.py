#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re
from typing import Any, Dict, Mapping, Tuple, Union

import torch
from pytext.common.constants import Padding, VocabMeta
from pytext.config.field_config import EmbedInitStrategy
from pytext.utils import data as data_utils, precision
from pytext.utils.label import get_label_weights
from torchtext import data as textdata
from torchtext.vocab import Vocab


def create_fields(fields_config, field_cls_dict):
    fields_config_dict = fields_config._asdict()
    return {
        name: field_cls.from_config(fields_config_dict[name])
        for name, field_cls in field_cls_dict.items()
        if fields_config_dict[name]
    }


def create_label_fields(label_configs, label_cls_dict):
    if not isinstance(label_configs, list):
        label_configs = [label_configs]
    labels: Dict[str, Field] = {}
    for label_config in label_configs:
        labels[label_config._name] = label_cls_dict[label_config._name].from_config(
            label_config
        )
    return labels


class FieldMeta:
    vocab: Vocab
    vocab_size: int
    vocab_export_name: str
    pad_token_idx: int
    unk_token_idx: int
    init_token_idx: int
    eos_token_idx: int
    nesting_meta: Any
    dummy_model_input: Union[torch.Tensor, Tuple[torch.Tensor, ...], None]


class Field(textdata.Field):
    @classmethod
    def from_config(cls, config):
        print(f"creating field {cls.__name__}")
        return cls(**config._asdict())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_meta(self) -> FieldMeta:
        meta = FieldMeta()
        if self.use_vocab:
            meta.vocab_size = len(self.vocab)
            meta.vocab = self.vocab
            if self.pad_token is not None:
                meta.pad_token_idx = self.vocab.stoi[self.pad_token]
            if self.unk_token is not None:
                meta.unk_token_idx = self.vocab.stoi[self.unk_token]
            if self.init_token is not None:
                meta.init_token_idx = self.vocab.stoi[self.init_token]
            if self.eos_token is not None:
                meta.eos_token_idx = self.vocab.stoi[self.eos_token]
        if hasattr(self, "dummy_model_input"):
            meta.dummy_model_input = self.dummy_model_input
        return meta

    def load_meta(self, metadata: FieldMeta):
        self.vocab = metadata.vocab

    def pad_length(self, n):
        """
        Override to make pad_length to be multiple of 8 to support fp16 training
        """
        return precision.pad_length(n)


class NestedField(Field, textdata.NestedField):
    def get_meta(self):
        meta = super().get_meta()
        meta.nesting_meta = self.nesting_field.get_meta()
        return meta

    def load_meta(self, metadata: FieldMeta):
        super().load_meta(metadata)
        self.nesting_field.vocab = metadata.nesting_meta.vocab


class RawField(textdata.RawField):
    def __init__(self, *args, is_target=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_target = is_target

    def get_meta(self) -> FieldMeta:
        return FieldMeta()


class VocabUsingField(Field):
    """Base class for all fields that need to build a vocabulary."""

    def __init__(
        self,
        pretrained_embeddings_path="",
        embed_dim=0,
        embedding_init_strategy=EmbedInitStrategy.RANDOM,
        vocab_file="",
        vocab_size="",
        vocab_from_train_data=True,  # add tokens from train data to vocab
        vocab_from_all_data=False,  # add tokens from train, eval, test data to vocab
        # add tokens from pretrained embeddings to vocab
        vocab_from_pretrained_embeddings=False,
        min_freq=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pretrained_embeddings_path = pretrained_embeddings_path
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.vocab_from_train_data = vocab_from_train_data
        self.vocab_from_all_data = vocab_from_all_data
        self.vocab_from_pretrained_embeddings = vocab_from_pretrained_embeddings
        self.min_freq = min_freq
        self.embed_dim = embed_dim
        self.embedding_init_strategy = embedding_init_strategy


class VocabUsingNestedField(VocabUsingField, NestedField):
    """Base class for all nested fields that need to build a vocabulary."""

    pass


class DocLabelField(Field):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            sequential=False,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
            unk_token=None,  # Don't include unk in the list of labels
        )


class WordLabelField(Field):
    def __init__(self, use_bio_labels, **kwargs):
        super().__init__(
            sequential=True,
            batch_first=True,
            tokenize=data_utils.simple_tokenize,
            pad_token=Padding.WORD_LABEL_PAD,
            unk_token=None,  # Don't include unk in the list of labels
        )
        self.use_bio_labels = use_bio_labels

    def get_meta(self):
        meta = super().get_meta()
        meta.use_bio_labels = self.use_bio_labels
        return meta


class TextFeatureField(VocabUsingField):
    dummy_model_input = torch.tensor([[1], [1]], dtype=torch.long, device="cpu")

    def __init__(
        self,
        pretrained_embeddings_path="",
        embed_dim=0,
        embedding_init_strategy=EmbedInitStrategy.RANDOM,
        vocab_file="",
        vocab_size="",
        vocab_from_train_data=True,
        vocab_from_all_data=False,
        vocab_from_pretrained_embeddings=False,
        postprocessing=None,
        use_vocab=True,
        include_lengths=True,
        batch_first=True,
        sequential=True,
        pad_token=VocabMeta.PAD_TOKEN,
        unk_token=VocabMeta.UNK_TOKEN,
        init_token=None,
        eos_token=None,
        lower=False,
        tokenize=data_utils.no_tokenize,
        fix_length=None,
        pad_first=None,
        min_freq=1,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_embeddings_path=pretrained_embeddings_path,
            embed_dim=embed_dim,
            embedding_init_strategy=embedding_init_strategy,
            vocab_file=vocab_file,
            vocab_size=vocab_size,
            vocab_from_train_data=vocab_from_train_data,
            vocab_from_all_data=vocab_from_all_data,
            vocab_from_pretrained_embeddings=vocab_from_pretrained_embeddings,
            postprocessing=postprocessing,
            use_vocab=use_vocab,
            include_lengths=include_lengths,
            batch_first=batch_first,
            sequential=sequential,
            pad_token=pad_token,
            unk_token=unk_token,
            init_token=init_token,
            eos_token=eos_token,
            lower=lower,
            tokenize=tokenize,
            fix_length=fix_length,
            pad_first=pad_first,
            min_freq=min_freq,
        )


class SeqFeatureField(VocabUsingNestedField):
    dummy_model_input = torch.tensor([[[1]], [[1]]], dtype=torch.long, device="cpu")

    def __init__(
        self,
        pretrained_embeddings_path="",
        embed_dim=0,
        embedding_init_strategy=EmbedInitStrategy.RANDOM,
        vocab_file="",
        vocab_size="",
        vocab_from_train_data=True,
        vocab_from_all_data=False,
        vocab_from_pretrained_embeddings=False,
        postprocessing=None,
        use_vocab=True,
        include_lengths=True,
        pad_token=VocabMeta.PAD_SEQ,
        init_token=None,
        eos_token=None,
        tokenize=data_utils.no_tokenize,
        nesting_field=None,
        **kwargs,
    ):
        super().__init__(
            pretrained_embeddings_path=pretrained_embeddings_path,
            embed_dim=embed_dim,
            embedding_init_strategy=embedding_init_strategy,
            vocab_file=vocab_file,
            vocab_size=vocab_size,
            vocab_from_train_data=vocab_from_train_data,
            vocab_from_all_data=vocab_from_all_data,
            vocab_from_pretrained_embeddings=vocab_from_pretrained_embeddings,
            postprocessing=postprocessing,
            use_vocab=use_vocab,
            include_lengths=include_lengths,
            pad_token=pad_token,
            init_token=init_token,
            eos_token=eos_token,
            tokenize=tokenize,
            nesting_field=nesting_field
            if nesting_field is not None
            else TextFeatureField(include_lengths=False, lower=False),
        )


class FloatField(Field):
    def __init__(self, **kwargs):
        super().__init__(
            sequential=False,
            use_vocab=False,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
            dtype=torch.float,
            unk_token=None,
        )


class FloatVectorField(Field):
    def __init__(self, dim=0, dim_error_check=False, **kwargs):
        super().__init__(
            use_vocab=False,
            batch_first=True,
            tokenize=self._parse_vector,
            dtype=torch.float,
            preprocessing=textdata.Pipeline(
                float
            ),  # Convert each string to float. float() takes care of whitespace.
            fix_length=dim,
            pad_token=0,  # For irregular sized vectors, pad the missing units with 0s.
        )
        self.dim_error_check = dim_error_check  # dims in data should match config
        self.dummy_model_input = torch.tensor(
            [[1.0] * dim, [1.0] * dim], dtype=torch.float, device="cpu"
        )

    def _parse_vector(self, s):
        if not s:
            return None
        if s[0] == "[":
            s = s[1:]  # Remove '['
        if s[-1] == "]":
            s = s[:-1]  # Remove ']'
        s_array = re.split("[, ]", s)  # Split on ',' or ' '
        ret_val = [s for s in s_array if s]  # Filter non strings
        assert not self.dim_error_check or len(ret_val) == self.fix_length, (
            "FloatVectorField has mismatched dimensions. Config dims:"
            + str(self.fix_length)
            + " data:"
            + str(len(ret_val))
            + ", actual vector:"
            + str(ret_val)
        )
        return ret_val


class ActionField(VocabUsingField):
    def __init__(self, **kwargs):
        super().__init__(
            use_vocab=True,
            sequential=True,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
            unk_token=None,  # Don't include UNK in the list of labels
            pad_token=None,  # Don't include PAD in the list of labels
            vocab_from_all_data=True,  # Actions must be built from all data.
        )
