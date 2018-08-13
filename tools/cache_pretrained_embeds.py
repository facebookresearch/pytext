#!/usr/bin/env python3

import os
import argparse
from pytext.utils.embeddings_utils import PretrainedEmbedding


def load_and_cache_embeddings(embeds_path: str, cache_path: str) -> None:
    if embeds_path is None or not os.path.isfile(embeds_path):
        raise ValueError("Invalid embeddings path")

    embeddings = PretrainedEmbedding()
    embeddings.load_pretrained_embeddings(embeds_path)
    embeddings.cache_pretrained_embeddings(cache_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_file",
        default="pytext/tests/data/pretrained_embed_raw",
        type=str,
    )
    parser.add_argument(
        "--embeddings_cache",
        default="pytext/tests/data/test_embed.cache",
        type=str,
    )
    args = parser.parse_args()
    load_and_cache_embeddings(args.embeddings_file, args.embeddings_cache)
