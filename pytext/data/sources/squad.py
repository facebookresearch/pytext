#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import math
from typing import List, Optional

from pytext.data.sources.data_source import (
    DataSource,
    SafeFileWrapper,
    generator_property,
)
from pytext.data.sources.tsv import TSV
from pytext.utils.path import get_absolute_path


def _shift_answers(orig_starts, piece_start, piece_end):
    # Re-align answer index for each piece when we split a long document.
    answer_starts = []
    has_answer = False
    for start in orig_starts:
        if start >= piece_start and start < piece_end:
            answer_starts.append(start - piece_start)
            has_answer = True
    return answer_starts, has_answer


def _split_document(
    id,
    doc,
    question,
    answers,
    answer_starts,
    has_answer,
    ignore_impossible,
    max_character_length,
    min_overlap,
):
    pieces = []
    min_overlap = math.floor(max_character_length * min_overlap)
    if has_answer or not ignore_impossible:
        n_pieces = 1 + math.ceil(
            max(0, len(doc) - max_character_length)
            / (max_character_length - min_overlap)
        )
        overlap = (
            math.floor((n_pieces * max_character_length - len(doc)) / (n_pieces - 1))
            if n_pieces > 1
            else 0
        )
        for n in range(n_pieces):
            start, end = (
                n * (max_character_length - overlap),
                (n + 1) * (max_character_length - overlap) + overlap,
            )
            answer_starts, piece_has_answer = _shift_answers(answer_starts, start, end)
            pieces.append(
                {
                    "id": id,
                    "doc": doc[start:end],
                    "question": question,
                    "answers": answers,
                    "answer_starts": answer_starts,
                    "has_answer": str(has_answer and piece_has_answer),
                }
            )
    return pieces


def process_squad_json(fname, ignore_impossible, max_character_length, min_overlap):
    if not fname:
        return
    with open(fname) as infile:
        dump = json.load(infile)

    id = 0
    for article in dump["data"]:
        for paragraph in article["paragraphs"]:
            doc = paragraph["context"]
            for question in paragraph["qas"]:
                has_answer = not question.get("is_impossible", False)
                answers = (
                    question["answers"] if has_answer else question["plausible_answers"]
                )
                question = question["question"]
                answer_texts = [answer["text"] for answer in answers]
                answer_starts = [int(answer["answer_start"]) for answer in answers]
                for piece_dict in _split_document(
                    id,
                    doc,
                    question,
                    answer_texts,
                    answer_starts,
                    has_answer,
                    ignore_impossible,
                    max_character_length,
                    min_overlap,
                ):
                    yield piece_dict
                id += 1


def process_squad_tsv(
    fname, ignore_impossible, max_character_length, min_overlap, delimiter, quoted
):
    if not fname:
        print(f"Empty file name!")
        return

    field_names = ["doc", "question", "answers", "answer_starts", "has_answer"]
    tsv_file = SafeFileWrapper(
        get_absolute_path(fname), encoding="utf-8", errors="replace"
    )
    tsv = TSV(
        tsv_file,
        field_names=field_names,
        delimiter=delimiter,
        quoted=quoted,
        drop_incomplete_rows=True,
    )

    for id, row in enumerate(tsv):
        doc, question, answers, answer_starts, has_answer = (
            row[f] for f in field_names
        )
        answers = json.loads(answers)
        answer_starts = json.loads(answer_starts)
        for piece_dict in _split_document(
            id,
            doc,
            question,
            answers,
            answer_starts,
            has_answer == "True",
            ignore_impossible,
            max_character_length,
            min_overlap,
        ):
            yield piece_dict


def process_squad(
    fname,
    ignore_impossible,
    max_character_length,
    min_overlap=0.1,
    delimiter="\t",
    quoted=False,
):
    if fname.split(".")[-1] == "json":
        return process_squad_json(
            fname, ignore_impossible, max_character_length, min_overlap
        )
    else:
        return process_squad_tsv(
            fname,
            ignore_impossible,
            max_character_length,
            min_overlap,
            delimiter,
            quoted,
        )


class SquadDataSource(DataSource):
    """
    Download data from https://rajpurkar.github.io/SQuAD-explorer/
    Will return tuples of (doc, question, answer, answer_start, has_answer)
    """

    class Config(DataSource.Config):
        train_filename: Optional[str] = "train-v2.0.json"
        test_filename: Optional[str] = "dev-v2.0.json"
        eval_filename: Optional[str] = "dev-v2.0.json"
        ignore_impossible: bool = True
        max_character_length: int = 2 ** 20
        min_overlap: float = 0.1  # Expressed as a fraction of the max_character_length.
        delimiter: str = "\t"
        quoted: bool = False

    @classmethod
    def from_config(cls, config: Config, schema=None):
        return cls(
            config.train_filename,
            config.test_filename,
            config.eval_filename,
            config.ignore_impossible,
            config.max_character_length,
            config.min_overlap,
            config.delimiter,
            config.quoted,
        )

    def __init__(
        self,
        train_filename=None,
        test_filename=None,
        eval_filename=None,
        ignore_impossible=Config.ignore_impossible,
        max_character_length=Config.max_character_length,
        min_overlap=Config.min_overlap,
        delimiter=Config.delimiter,
        quoted=Config.quoted,
    ):
        schema = {
            "id": int,
            "doc": str,
            "question": str,
            "answers": List[str],
            "answer_starts": List[int],
            "answer_ends": List[int],
            "has_answer": str,
        }
        super().__init__(schema)
        self.train_filename = train_filename
        self.test_filename = test_filename
        self.eval_filename = eval_filename
        self.ignore_impossible = ignore_impossible
        self.max_character_length = max_character_length
        self.min_overlap = min_overlap
        self.delimiter = delimiter
        self.quoted = quoted

    def process_file(self, fname):
        return process_squad(
            fname,
            self.ignore_impossible,
            self.max_character_length,
            self.min_overlap,
            self.delimiter,
            self.quoted,
        )

    @generator_property
    def train(self):
        return self.process_file(self.train_filename)

    @generator_property
    def test(self):
        return self.process_file(self.test_filename)

    @generator_property
    def eval(self):
        return self.process_file(self.eval_filename)
