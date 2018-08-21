#!/usr/bin/env python3
from typing import Any, Dict

import torch
from pytext.common.constants import Padding, VocabMeta
from pytext.utils import data_utils
from torchtext import data as textdata


class Field(textdata.Field):
    def __init__(self, name, export_input_names=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.export_input_names = export_input_names or [name]

    def get_meta(self) -> Dict[str, Any]:
        return {}


class RawField(textdata.RawField):
    def __init__(self, name, export_input_names=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.export_input_names = export_input_names or [name]

    def get_meta(self) -> Dict[str, Any]:
        return {}


class DocLabelField(Field):
    def __init__(self, name):
        super().__init__(
            name,
            sequential=False,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
            unk_token=None,  # Don't include unk in the list of labels
        )

    def get_meta(self) -> Dict[str, Any]:
        return {"doc_class_num": len(self.vocab)}


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

    def get_meta(self) -> Dict[str, Any]:
        return {"word_class_num": len(self.vocab)}


class TextFeatureField(Field):
    def __init__(
        self,
        name,
        export_input_names=None,
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
    ):
        super().__init__(
            name,
            export_input_names,
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

    def get_meta(self) -> Dict[str, Any]:
        meta = {
            "embed_num": len(self.vocab),
            "pad_idx": self.vocab.stoi[VocabMeta.PAD_TOKEN],
            "unk_idx": self.vocab.stoi[VocabMeta.UNK_TOKEN],
        }
        if self.init_token is not None:
            meta["init_token_idx"] = self.vocab.stoi[VocabMeta.INIT_TOKEN]

        if self.eos_token is not None:
            meta["eos_token_idx"] = self.vocab.stoi[VocabMeta.EOS_TOKEN]

        return meta


class CapFeatureField(Field):
    def __init__(self, name, export_input_names=None):
        super().__init__(
            name,
            export_input_names,
            use_vocab=False,
            sequential=True,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
        )

    def get_meta(self) -> Dict[str, Any]:
        # TODO replace with const in featurizer
        return {"cap_embed_num": 6}


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
