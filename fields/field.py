#!/usr/bin/env python3

from typing import Any, Mapping

import torch
from pytext.common.constants import Padding, VocabMeta
from pytext.config.field_config import EmbedInitStrategy
from pytext.fields.utils import reverse_tensor
from pytext.utils import data_utils
from torchtext import data as textdata
from torchtext.vocab import Vocab


def create_fields(fields_config, field_cls_dict):
    return {
        name: field_cls_dict[name].from_config(field_config)
        for name, field_config in fields_config.items()
        if field_config
    }


class FieldMeta:
    vocab: Vocab
    vocab_size: int
    vocab_export_name: str
    pad_token_idx: int
    unk_token_idx: int
    init_token_idx: int
    eos_token_idx: int
    nesting_meta: Any


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
        return meta

    def load_meta(self, metadata: FieldMeta):
        self.vocab = metadata.vocab


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
        vocab_from_train_data=True,
        vocab_from_all_data=False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pretrained_embeddings_path = pretrained_embeddings_path
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.vocab_from_train_data = vocab_from_train_data
        self.vocab_from_all_data = vocab_from_all_data
        self.embed_dim = embed_dim
        self.embedding_init_strategy = embedding_init_strategy


class VocabUsingNestedField(VocabUsingField, NestedField):
    """Base class for all nested fields that need to build a vocabulary."""

    pass


class DocLabelField(Field):
    def __init__(self, label_weights: Mapping[str, float] = None, **kwargs) -> None:
        super().__init__(
            sequential=False,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
            unk_token=None,  # Don't include unk in the list of labels
        )
        self.label_weights = label_weights or {}

    # TODO: Create LabelFieldMeta class
    def get_meta(self):
        meta = super().get_meta()
        # Now that the vocabulary has been built, prune the label_weights to
        # remove the labels that do not exist in the dataset
        pruned_label_weights = {
            meta.vocab.stoi[k]: v
            for (k, v) in self.label_weights.items()
            if k in meta.vocab.stoi
        }
        if len(pruned_label_weights) == 0:
            return meta

        # All unspecified classes will get a weight of 1
        meta.label_weights = torch.ones(meta.vocab_size)
        for k, v in pruned_label_weights.items():
            meta.label_weights[k] = v
        # Create the weight tensor on the right device
        meta.label_weights = meta.label_weights.numpy()
        return meta


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
    def __init__(
        self,
        pretrained_embeddings_path="",
        embed_dim=0,
        embedding_init_strategy=EmbedInitStrategy.RANDOM,
        vocab_file="",
        vocab_size="",
        vocab_from_train_data=True,
        postprocessing=None,
        use_vocab=True,
        include_lengths=True,
        batch_first=True,
        sequential=True,
        pad_token=VocabMeta.PAD_TOKEN,
        unk_token=VocabMeta.UNK_TOKEN,
        init_token=None,
        eos_token=None,
        lower=True,
        tokenize=data_utils.no_tokenize,
        fix_length=None,
        **kwargs
    ) -> None:
        super().__init__(
            pretrained_embeddings_path=pretrained_embeddings_path,
            embed_dim=embed_dim,
            embedding_init_strategy=embedding_init_strategy,
            vocab_file=vocab_file,
            vocab_size=vocab_size,
            vocab_from_train_data=vocab_from_train_data,
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
        )


class SeqFeatureField(VocabUsingNestedField):
    def __init__(
        self,
        pretrained_embeddings_path="",
        embed_dim=0,
        embedding_init_strategy=EmbedInitStrategy.RANDOM,
        vocab_file="",
        vocab_size="",
        vocab_from_train_data=True,
        postprocessing=None,
        use_vocab=True,
        include_lengths=True,
        pad_token=VocabMeta.PAD_SEQ,
        init_token=None,
        eos_token=None,
        tokenize=data_utils.no_tokenize,
        nesting_field=None,
        **kwargs
    ):
        super().__init__(
            pretrained_embeddings_path=pretrained_embeddings_path,
            embed_dim=embed_dim,
            embedding_init_strategy=embedding_init_strategy,
            vocab_file=vocab_file,
            vocab_size=vocab_size,
            vocab_from_train_data=vocab_from_train_data,
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


class ActionField(VocabUsingField):
    def __init__(self, **kwargs):
        super().__init__(
            use_vocab=True,
            sequential=True,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
            unk_token=None,  # Don't include unk in the list of labels
            # reverse the tensor
            postprocessing=reverse_tensor,
        )
