#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import threading
from typing import Dict, List, Optional, Type

from pytext.data.sources.data_source import RootDataSource, SafeFileWrapper


class CoNLLUPOSDataSource(RootDataSource):
    """DataSource which loads data from CoNLL-U file."""

    class Config(RootDataSource.Config):
        #: Name of the language. If not set, languages will be empty.
        language: Optional[str] = None
        #: Filename of training set. If not set, iteration will be empty.
        train_filename: Optional[str] = None
        #: Filename of testing set. If not set, iteration will be empty.
        test_filename: Optional[str] = None
        #: Filename of eval set. If not set, iteration will be empty.
        eval_filename: Optional[str] = None
        #: Field names for the TSV. If this is not set, the first line of each file
        #: will be assumed to be a header containing the field names.
        field_names: Optional[List[str]] = None
        #: The column delimiter. CoNLL-U file default is \t.
        delimiter: str = "\t"

    @classmethod
    def from_config(cls, config: Config, schema: Dict[str, Type], **kwargs):
        args = config._asdict()
        language = args.pop("language")
        train_filename = args.pop("train_filename")
        test_filename = args.pop("test_filename")
        eval_filename = args.pop("eval_filename")
        train_file = (
            SafeFileWrapper(train_filename, encoding="utf-8", errors="replace")
            if train_filename
            else None
        )
        test_file = (
            SafeFileWrapper(test_filename, encoding="utf-8", errors="replace")
            if test_filename
            else None
        )
        eval_file = (
            SafeFileWrapper(eval_filename, encoding="utf-8", errors="replace")
            if eval_filename
            else None
        )
        return cls(
            language=language,
            train_file=train_file,
            test_file=test_file,
            eval_file=eval_file,
            schema=schema,
            **args,
            **kwargs,
        )

    def __init__(
        self,
        language=None,
        train_file=None,
        test_file=None,
        eval_file=None,
        field_names=None,
        delimiter=Config.delimiter,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.delimiter = delimiter
        self.language = language
        self._train_file = self._read_file(train_file) if train_file else []
        self._test_file = self._read_file(test_file) if test_file else []
        self._eval_file = self._read_file(eval_file) if eval_file else []

    def _read_file(self, input_file):
        """Reads CoNLL-U file"""
        words, labels = [], []
        for line in input_file.readlines():
            tok = line.strip().split(self.delimiter)
            # skip comment and empty line, yield if we got a sentence
            # CoNLL-U file separates sentences with empty line
            if len(tok) < 2 or line[0] == "#":
                assert len(words) == len(labels)
                if words:
                    yield {"text": words, "label": labels, "language": self.language}
                    words, labels = [], []
            elif tok[0].isdigit():
                word, pos = tok[1], tok[3]
                words.append(word)
                labels.append(pos)
        if len(words) == len(labels) and words:
            yield {"text": words, "label": labels, "language": self.language}

    def raw_train_data_generator(self):
        return iter(self._train_file)

    def raw_test_data_generator(self):
        return iter(self._test_file)

    def raw_eval_data_generator(self):
        return iter(self._eval_file)


class CoNLLUNERFile:
    def __init__(self, file, delim, lang):
        self.file = file
        self.delimiter = delim
        self.language = lang
        self._access_lock = threading.Lock()

    def __iter__(self):
        can_acquire = self._access_lock.acquire(blocking=False)
        if not can_acquire:
            raise Exception("Concurrent iteration not supported")
        self.file.seek(0)
        try:
            words, labels = [], []
            for line in self.file.readlines():
                line = line.strip()
                tok = line.split(self.delimiter)
                if not line:
                    assert len(words) == len(labels)
                    if words:
                        yield {
                            "text": words,
                            "label": labels,
                            "language": self.language,
                        }
                        words, labels = [], []
                elif len(tok) == 2:
                    word, label = tok
                    words.append(word)
                    labels.append(label)
            if len(words) == len(labels) and words:
                yield {"text": words, "label": labels, "language": self.language}
        finally:
            self._access_lock.release()


class CoNLLUNERDataSource(CoNLLUPOSDataSource):
    """
    Reads an empty line separated data (word \t label).
    This data source supports datasets for NER tasks
    """

    def _read_file(self, input_file):
        return CoNLLUNERFile(input_file, self.delimiter, self.language)
