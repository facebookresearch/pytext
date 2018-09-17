#!/usr/bin/env python3

import torch
from pytext.common.constants import Padding, VocabMeta
from pytext.config.field_config import EmbedInitStrategy
from pytext.fields.utils import reverse_tensor
from pytext.utils import data_utils
from torchtext import data as textdata
from torchtext.vocab import Vocab


class FieldMeta:
    vocab: Vocab
    vocab_size: int
    vocab_export_name: str
    pad_token_idx: int
    unk_token_idx: int
    init_token_idx: int
    eos_token_idx: int


class Field(textdata.Field):
    def __init__(self, name, export_names=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.export_names = export_names or [name]

    def get_meta(self) -> FieldMeta:
        meta = FieldMeta()
        if self.use_vocab:
            meta.vocab_size = len(self.vocab)
            meta.vocab = self.vocab
            meta.vocab_export_name = self.export_names[0]
        if self.pad_token is not None:
            meta.pad_token_idx = self.vocab.stoi[self.pad_token]
        if self.unk_token is not None:
            meta.unk_token_idx = self.vocab.stoi[self.unk_token]
        if self.init_token is not None:
            meta.init_token_idx = self.vocab.stoi[self.init_token]
        if self.eos_token is not None:
            meta.eos_token_idx = self.vocab.stoi[self.eos_token]
        return meta


class RawField(textdata.RawField):
    def __init__(self, name, export_names=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.export_names = export_names or [name]


class VocabUsingField(Field):
    """Base class for all fields that need to build a vocabulary."""

    def __init__(
        self,
        *args,
        pretrained_embeddings_path="",
        embed_dim=0,
        embedding_init_strategy=EmbedInitStrategy.RANDOM,
        vocab_file="",
        vocab_size="",
        vocab_from_train_data=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pretrained_embeddings_path = pretrained_embeddings_path
        self.vocab_file = vocab_file
        self.vocab_size = vocab_size
        self.vocab_from_train_data = vocab_from_train_data
        self.embed_dim = embed_dim
        self.embedding_init_strategy = embedding_init_strategy


class DocLabelField(Field):
    def __init__(self, name):
        super().__init__(
            name,
            sequential=False,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
            unk_token=None,  # Don't include unk in the list of labels
        )


class WordLabelField(Field):
    def __init__(self, name, use_bio_labels):
        super().__init__(
            name,
            sequential=True,
            batch_first=True,
            tokenize=data_utils.simple_tokenize,
            pad_token=Padding.WORD_LABEL_PAD,
            unk_token=None,  # Don't include unk in the list of labels
        )
        self.use_bio_labels = use_bio_labels


class TextFeatureField(VocabUsingField):
    def __init__(
        self,
        name,
        export_names=None,
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
    ) -> None:
        super().__init__(
            name,
            export_names,
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
        )


class CapFeatureField(Field):
    def __init__(self, name, export_names=None):
        super().__init__(
            name,
            export_names,
            use_vocab=False,
            sequential=True,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
        )


class FloatField(Field):
    def __init__(self, name):
        super().__init__(
            name,
            sequential=False,
            use_vocab=False,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
            dtype=torch.float,
            unk_token=None,
        )


class ActionField(VocabUsingField):
    def __init__(self, name):
        super().__init__(
            name,
            use_vocab=True,
            sequential=True,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
            unk_token=None,  # Don't include unk in the list of labels
            # reverse the tensor
            postprocessing=reverse_tensor,
        )
