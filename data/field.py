#!/usr/bin/env python3
from typing import Any, Dict

import torch
from pytext.common.constants import Padding, VocabMeta
from pytext.custom_fields.char_field import CharField
from pytext.custom_fields.dict_field import DictFeatField
from pytext.custom_fields.text_field import TextField
from pytext.utils import data_utils
from torchtext import data as textdata


# TODO T31514507 just inherit from torchtext Field class once we decide to embrace torchtext
class Field:
    def __init__(self, name, export_input_names=None, field=None):
        self.name = name
        self.export_input_names = export_input_names or [name]
        self.field: textdata.Field = field

    def get_meta(self) -> Dict[str, Any]:
        return {}


class RawField(Field):
    def __init__(self, name):
        super().__init__(name)
        self.field = textdata.RawField()


class DocLabelField(Field):
    def __init__(self, name):
        super().__init__(name)
        self.field = textdata.Field(
            sequential=False,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
            unk_token=None,  # Don't include unk in the list of labels
        )

    def get_meta(self) -> Dict[str, Any]:
        return {"doc_class_num": len(self.field.vocab)}


class WordLabelField(Field):
    def __init__(self, name, use_bio_labels):
        super().__init__(name)
        self.use_bio_labels = use_bio_labels
        self.field = textdata.Field(
            sequential=True,
            batch_first=True,
            tokenize=data_utils.simple_tokenize,
            pad_token=Padding.WORD_LABEL_PAD,
            unk_token=None,  # Don't include unk in the list of labels
        )

    def get_meta(self) -> Dict[str, Any]:
        return {"word_class_num": len(self.field.vocab)}


class TextFeature(Field):
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
        super().__init__(name, export_input_names)
        # TODO: using custom field TextField because texdata.field does
        # not allow the passing of existing vocabulary to build_vocab
        self.field = TextField(
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
            "embed_num": len(self.field.vocab),
            "pad_idx": self.field.vocab.stoi[VocabMeta.PAD_TOKEN],
            "unk_idx": self.field.vocab.stoi[VocabMeta.UNK_TOKEN],
        }
        if self.field.init_token is not None:
            meta["init_token_idx"] = self.field.vocab.stoi[VocabMeta.INIT_TOKEN]

        if self.field.eos_token is not None:
            meta["eos_token_idx"] = self.field.vocab.stoi[VocabMeta.EOS_TOKEN]

        return meta


class DictFeature(Field):
    def __init__(self, name, export_input_names=None):
        super().__init__(name, export_input_names)
        self.field = DictFeatField(
            VocabMeta.PAD_TOKEN, VocabMeta.UNK_TOKEN, batch_first=True
        )

    def get_meta(self) -> Dict[str, Any]:
        return {"dict_embed_num": len(self.field.vocab)}


class CharFeature(Field):
    def __init__(self, name):
        super().__init__(name)
        self.field = CharField(
            VocabMeta.PAD_TOKEN, VocabMeta.UNK_TOKEN, batch_first=True
        )

    def get_meta(self) -> Dict[str, Any]:
        return {"char_embed_num": len(self.field.vocab)}


class CapFeature(Field):
    def __init__(self, name):
        super().__init__(name)
        self.field = textdata.Field(
            use_vocab=False,
            sequential=True,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
        )

    def get_meta(self) -> Dict[str, Any]:
        # TODO replace with const in featurizer
        return {"cap_embed_num": 6}


class LossWeightField(Field):
    def __init__(self, name):
        super().__init__(name)
        self.field = textdata.Field(
            sequential=False,
            use_vocab=False,
            batch_first=True,
            tokenize=data_utils.no_tokenize,
            dtype=torch.float,
            unk_token=None,
        )
