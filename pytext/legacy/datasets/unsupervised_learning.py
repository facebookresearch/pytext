#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

import torch
from torchtext.data.functional import custom_replace
from torchtext.data.functional import simple_space_split
from torchtext.utils import download_from_url, extract_archive

from ..vocab import build_vocab_from_iterator


_patterns = [
    (r"<.*>", ""),
    (r"&amp;", "&"),
    (r"&lt;", "<"),
    (r"&gt;", ">"),
    (r"<ref[^<]*<\/ref>", ""),
    (r"<[^>]*>", ""),
    (r"\[http:[^] ]*", "["),
    (r"\|thumb", ""),
    (r"\|left", ""),
    (r"\|right", ""),
    (r"\|\d+px", ""),
    (r"\[\[image:[^\[\]]*\|", ""),
    (r"\[\[category:([^|\]]*)[^]]*\]\]", "[[$1]]"),
    (r"\[\[[a-z\-]*:[^\]]*\]\]", ""),
    (r"\[\[[^\|\]]*\|", "[["),
    (r"\{\{[^\}]*\}\}", ""),
    (r"\{[^\}]*\}", ""),
    (r"\[", ""),
    (r"\]", ""),
    (r"&[^;]*;", " "),
    (r"A", "a"),
    (r"B", "b"),
    (r"C", "c"),
    (r"D", "d"),
    (r"E", "e"),
    (r"F", "f"),
    (r"G", "g"),
    (r"H", "h"),
    (r"I", "i"),
    (r"J", "j"),
    (r"K", "k"),
    (r"L", "l"),
    (r"M", "m"),
    (r"N", "n"),
    (r"O", "o"),
    (r"P", "p"),
    (r"Q", "q"),
    (r"R", "r"),
    (r"S", "s"),
    (r"T", "t"),
    (r"U", "u"),
    (r"V", "v"),
    (r"W", "w"),
    (r"X", "x"),
    (r"Y", "y"),
    (r"Z", "z"),
    (r"0", " zero "),
    (r"1", " one "),
    (r"2", " two "),
    (r"3", " three "),
    (r"4", " four "),
    (r"5", " five "),
    (r"6", " six "),
    (r"7", " seven "),
    (r"8", " eight "),
    (r"9", " nine "),
    (r"[^a-z\n]+", " "),
    (r"\n ", ""),
    (r"\s+", " "),
    (r"\n\s*\n", r"\n"),
]
enwik9_norm_transform = custom_replace(_patterns)


def generate_offsets(filename):
    offsets = []
    with open(filename) as f:
        offsets.append(f.tell())
        while f.readline():
            offsets.append(f.tell())
    return offsets


def read_lines_from_iterator(data_path, offsets, begin_line, num_lines):
    with open(data_path) as f:
        f.seek(offsets[begin_line])
        for _ in range(num_lines):
            yield f.readline()


def preprocess_raw_enwik9(input_filename, output_filename):
    with open(input_filename, "r") as f1:
        with open(output_filename, "w") as f2:
            while True:
                line = f1.readline()
                if not line:
                    break
                line = list(enwik9_norm_transform([line]))[0]
                if line != " " and line != "":
                    if line[0] == " ":
                        line = line[1:]
                    f2.writelines(line + "\n")


class EnWik9(torch.utils.data.Dataset):
    r"""Compressed size of first 10^9 bytes of enwiki-20060303-pages-articles.xml.
    It's part of Large Text Compression Benchmark project
    """

    def __init__(self, begin_line=0, num_lines=6348957, root=".data"):
        """Initiate EnWik9 dataset.

        Args:
            begin_line: the number of beginning line. Default: 0
            num_lines: the number of lines to be loaded. Default: 6348957
            root: Directory where the datasets are saved. Default: ".data"
            data: a list of label/tokens tuple. tokens are a tensor after

        Examples:
            >>> from torchtext.datasets import EnWik9
            >>> enwik9 = EnWik9(num_lines=20000)
            >>> vocab = enwik9.get_vocab()
        """

        super(EnWik9, self).__init__()

        processed_file = os.path.join(root, "norm_enwik9")
        if not os.path.exists(processed_file):
            url = "http://mattmahoney.net/dc/enwik9.zip"
            dataset_zip = download_from_url(
                url, path=os.path.join(root, "enwik9.zip"), root=root
            )
            extracted_file = extract_archive(dataset_zip)
            raw_file = extracted_file[0]
            preprocess_raw_enwik9(raw_file, processed_file)

        # Meta information
        offsets = generate_offsets(processed_file)
        read_lines = read_lines_from_iterator(
            processed_file, offsets, begin_line, num_lines
        )

        self._data = []
        for item in simple_space_split(read_lines):
            self._data += item

        self._vocab = None

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_vocab(self):
        if self._vocab is None:
            self._vocab = build_vocab_from_iterator([self._data])
        return self._vocab
