#!/usr/bin/env python3

import os
import argparse
from pytext.utils.embeddings_utils import PretrainedEmbedding


def load_and_concat_xlu_embeddings(source_files, target_file, target_raw_file=None):

    embeddings = PretrainedEmbedding()
    for source_file in source_files:
        dialect, embeds_path = source_file.split(":", 1)
        if embeds_path is None or not os.path.isfile(embeds_path):
            raise ValueError("Invalid embeddings path: {}", embeds_path)
        embeddings.load_pretrained_embeddings(embeds_path, True, dialect)
    embeddings.cache_pretrained_embeddings(target_file)

    # also save concatenated embeddings in raw form
    if target_raw_file is not None:
        with open(target_raw_file, "w") as out_file:
            print(
                len(embeddings.embed_vocab),
                len(embeddings.embedding_vectors[0]),
                file=out_file,
            )
            for i, word in enumerate(embeddings.embed_vocab):
                vector = embeddings.embedding_vectors[i]
                print(word, " ".join(map(str, vector.numpy())), file=out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings_files",
        required=True,
        type=str,
        nargs="+",
        help="a list of raw XLU embedding files in the form"
        "dialect:PATH, separated by spaces",
    )
    parser.add_argument("--embeddings_cache", required=True, type=str)
    parser.add_argument(
        "--embeddings_raw",
        required=False,
        type=str,
        help="Also save concatenated embeddings in raw form to this path",
    )

    args = parser.parse_args()
    load_and_concat_xlu_embeddings(
        args.embeddings_files, args.embeddings_cache, args.embeddings_raw
    )
