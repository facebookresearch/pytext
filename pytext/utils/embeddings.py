#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
from typing import Dict, List  # noqa: F401

import numpy as np
import torch
from pytext.common.constants import PackageFileName
from pytext.config.field_config import EmbedInitStrategy


class PretrainedEmbedding(object):
    """
    Utility class for loading/caching/initializing word embeddings
    """

    def __init__(
        self, embeddings_path: str = None, lowercase_tokens: bool = True
    ) -> None:
        if embeddings_path:
            if os.path.isdir(embeddings_path):
                serialized_embed_path = os.path.join(
                    embeddings_path, PackageFileName.SERIALIZED_EMBED
                )
                raw_embeddings_path = os.path.join(
                    embeddings_path, PackageFileName.RAW_EMBED
                )
            elif os.path.isfile(embeddings_path):
                serialized_embed_path = ""
                raw_embeddings_path = embeddings_path
            else:
                raise FileNotFoundError(
                    f"{embeddings_path} not found. Can't load pretrained embeddings."
                )

            if os.path.isfile(serialized_embed_path):
                try:
                    self.load_cached_embeddings(serialized_embed_path)
                except Exception:
                    print("Failed to load cached embeddings, loading the raw file.")
                    self.load_pretrained_embeddings(
                        raw_embeddings_path, lowercase_tokens=lowercase_tokens
                    )
            else:
                self.load_pretrained_embeddings(
                    raw_embeddings_path, lowercase_tokens=lowercase_tokens
                )
        else:
            self.embed_vocab = []  # type: List[str]
            self.stoi = {}  # type: Dict[str, int]
            self.embedding_vectors = None  # type: torch.Tensor

    def load_pretrained_embeddings(
        self,
        raw_embeddings_path: str,
        append: bool = False,
        dialect: str = None,
        lowercase_tokens: bool = True,
    ) -> None:
        """
        Loading raw embeddings vectors from file in the format:
        num_words dim
        word_i v0, v1, v2, ...., v_dim
        word_2 v0, v1, v2, ...., v_dim
        ....
        Optionally appends _dialect to every token in the vocabulary
        (for XLU embeddings).
        """
        chunk_vocab = []

        def iter_parser(
            skip_header: int = 0, delimiter: str = " ", dtype: type = np.float32
        ):
            """ Iterator to load numpy 1-d array from multi-row text file,
            where format is assumed to be:
                word_i v0, v1, v2, ...., v_dim
                word_2 v0, v1, v2, ...., v_dim
            The iterator will omit the first column (vocabulary) and via closure
            store values into the 'chunk_vocab' list.
            """
            tokens = set()
            with open(raw_embeddings_path, "r", errors="backslashreplace") as txtfile:
                for _ in range(skip_header):
                    next(txtfile)
                for line in txtfile:
                    split_line = line.rstrip("\r\n ").split(delimiter)
                    token = split_line[0]
                    if lowercase_tokens:
                        # lowercase here so that returned matrix doesn't contain
                        # the same token twice (lower and upper cases).
                        token = token.lower()
                    if token not in tokens:
                        chunk_vocab.append(token)
                        for item in split_line[1:]:
                            yield dtype(item)

        t = time.time()
        embed_array = np.fromiter(iter_parser(skip_header=1), dtype=np.float32)
        embed_matrix = embed_array.reshape((len(chunk_vocab), -1))
        print("Rows loaded: ", embed_matrix.shape[0], "; Time: ", time.time() - t, "s.")

        if not append:
            self.embed_vocab = []
            self.stoi = {}

        if dialect is not None:
            chunk_vocab = [append_dialect(word, dialect) for word in chunk_vocab]

        self.embed_vocab.extend(chunk_vocab)
        self.stoi = {word: i for i, word in enumerate(chunk_vocab)}

        if append and self.embedding_vectors is not None:
            self.embedding_vectors = torch.cat(
                (self.embedding_vectors, torch.Tensor(embed_matrix))
            )
        else:
            self.embedding_vectors = torch.Tensor(embed_matrix)

    def cache_pretrained_embeddings(self, cache_path: str) -> None:
        """
        Cache the processed embedding vectors and vocab to a file for faster
        loading
        """
        torch.save((self.embed_vocab, self.stoi, self.embedding_vectors), cache_path)

    def load_cached_embeddings(self, cache_path: str) -> None:
        """
        Load cached embeddings from file
        """
        self.embed_vocab, self.stoi, self.embedding_vectors = torch.load(cache_path)

    def initialize_embeddings_weights(
        self,
        str_to_idx: Dict[str, int],
        unk: str,
        embed_dim: int,
        init_strategy: EmbedInitStrategy,
    ) -> torch.Tensor:
        """
        Initialize embeddings weights of shape (len(str_to_idx), embed_dim) from the
        pretrained embeddings vectors. Words that are not in the pretrained
        embeddings list will be initialized according to `init_strategy`.
        :param str_to_idx: a dict that maps words to indices that the model expects
        :param unk: unknown token
        :param embed_dim: the embeddings dimension
        :param init_strategy: method of initializing new tokens
        :returns: a float tensor of dimension (vocab_size, embed_dim)
        """
        pretrained_embeds_weight = torch.Tensor(len(str_to_idx), embed_dim)

        if init_strategy == EmbedInitStrategy.RANDOM:
            pretrained_embeds_weight.normal_(0, 1)
        elif init_strategy == EmbedInitStrategy.ZERO:
            pretrained_embeds_weight.fill_(0.0)
        else:
            raise ValueError(
                "Unknown embedding initialization strategy '{}'".format(init_strategy)
            )

        assert self.embedding_vectors is not None and self.embed_vocab is not None
        assert pretrained_embeds_weight.shape[-1] == self.embedding_vectors.shape[-1]
        unk_idx = str_to_idx[unk]
        oov_count = 0
        for word, idx in str_to_idx.items():
            if word in self.stoi and idx != unk_idx:
                pretrained_embeds_weight[idx] = self.embedding_vectors[self.stoi[word]]
            else:
                oov_count += 1
        print(
            f"{oov_count}/{len(str_to_idx)} tokens were found out of "
            f"pretrained embedding vocabulary"
        )
        return pretrained_embeds_weight


def append_dialect(word: str, dialect: str) -> str:
    if word.endswith("-{}".format(dialect)):
        return word
    else:
        return f"{word}-{dialect}"
