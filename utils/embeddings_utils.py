#!/usr/bin/env python3
import csv
import os
import time
from typing import Dict, List  # noqa: F401

import numpy as np
import pandas
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
                    "f{embeddings_path} not found. Can't load pretrained embeddings."
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
        [word_i] [v0, v1, v2, ...., v_dim]
        [word_2] [v0, v1, v2, ...., v_dim]
        ....

        Optionally appends _dialect to every token in the vocabulary
        (for XLU embeddings).
        """
        filetype = {i: np.float32 for i in range(1, 10000)}
        filetype[0] = np.str
        embed_df = pandas.read_csv(
            raw_embeddings_path,
            chunksize=500000,
            skiprows=1,  # Assuming the file has a header in the first row
            delim_whitespace=True,
            dtype=filetype,
            header=None,
            na_filter=False,
            quoting=csv.QUOTE_NONE,
        )
        embed_vectors = []  # type: List[List[float]]

        if not append:
            self.embed_vocab = []
            self.stoi = {}
        t = time.time()
        size = 0
        for chunk in embed_df:
            chunk_vocab = chunk.as_matrix([0]).flatten()
            if lowercase_tokens:
                chunk_vocab = map(lambda s: s.lower(), chunk_vocab)
            if dialect is not None:
                chunk_vocab = [append_dialect(word, dialect) for word in chunk_vocab]
            self.embed_vocab.extend(chunk_vocab)
            for i, word in enumerate(chunk_vocab):
                self.stoi[word] = size + i
            embed_vectors.extend(chunk.as_matrix(list(range(1, len(chunk.columns)))))
            size += len(chunk)
            print("Rows loaded: ", size, "; Time: ", time.time() - t, "s.")

        if append and self.embedding_vectors is not None:
            new_vectors = torch.Tensor(embed_vectors).view(size, -1)
            self.embedding_vectors = embed_vectors = torch.cat(
                (self.embedding_vectors, new_vectors)
            )
        else:
            self.embedding_vectors = torch.Tensor(embed_vectors).view(
                len(self.embed_vocab), -1
            )

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
        vocab_to_idx: Dict[str, int],
        unk_idx: int,
        vocab_size: int,
        embed_dim: int,
        init_strategy: EmbedInitStrategy,
    ) -> torch.Tensor:
        """
        Initialize embeddings weights of shape (vocab_size, embed_dim) from the
        pretrained embeddings vectors. Words that are not in the pretrained
        embeddings list will be randomly initialized.
        :param vocab_to_idx: a dict that maps words to indices that the model expects
        :param unk_idx: index of unknown word
        :param vocab_size: the number of unique words in the model vocab
        :param embed_dim: the embeddings dimension
        :returns: a float tensor of dimension (vocab_size, embed_dim)
        """
        pretrained_embeds_weight = torch.Tensor(vocab_size, embed_dim)

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
        for word, idx in vocab_to_idx.items():
            if word in self.stoi and idx != unk_idx:
                pretrained_embeds_weight[idx] = self.embedding_vectors[self.stoi[word]]
        return pretrained_embeds_weight


def init_pretrained_embeddings(
    vocab_to_id: Dict[str, int],
    embeddings_path: str,
    embed_dim: int,
    unk: str,
    embedding_init_strategy: EmbedInitStrategy,
    lowercase_tokens: bool,
) -> torch.Tensor:
    return PretrainedEmbedding(
        embeddings_path, lowercase_tokens
    ).initialize_embeddings_weights(
        vocab_to_id,
        vocab_to_id[unk],
        len(vocab_to_id),
        embed_dim,
        embedding_init_strategy,
    )


def append_dialect(word: str, dialect: str) -> str:
    if word.endswith("-{}".format(dialect)):
        return word
    else:
        return f"{word}-{dialect}"
