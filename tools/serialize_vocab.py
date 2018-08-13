#!/usr/bin/env python3

import argparse
from typing import List, Set
from . import init_logger
from thrift.util import Serializer
from assistant.lib.feat.ttypes import Vocab, ModelVocabs
from thrift.protocol import TSimpleJSONProtocol

LOGGER = init_logger(logger_name="VocabSerializer")
PARSER = argparse.ArgumentParser(description="Process some integers.")
PARSER.add_argument(
    "-t", "--intokenvocabfile", required=True, help="input token vocab file"
)
PARSER.add_argument(
    "-d", "--indictvocabfile", required=True, help="input dict feat vocab file"
)
PARSER.add_argument(
    "-o", "--outfile", required=True, help="output serialized vocab file"
)


def load_vocab(file_path: str) -> Vocab:
    vocab_list: List[str] = []
    with open(file_path, "r") as f:
        LOGGER.info("Loading vocab from {}".format(file_path))
        vocab_set: Set[str] = set()
        for line in f:
            token = line.strip().split("\t")[0].strip()
            if token in vocab_set:
                LOGGER.warning(
                    """{} appeared at more than once in the file.
                    Ignoring the latter.""".format(
                        token
                    )
                )
            else:
                vocab_list.append(token)
                vocab_set.add(token)
    vocab_obj = Vocab()
    vocab_obj.vocab = vocab_list
    return vocab_obj


def serialize_model_vocabs(
    token_vocab: Vocab, dict_vocab: Vocab, out_file_path: str
) -> None:
    model_vocabs = ModelVocabs()
    model_vocabs.tokenVocab = token_vocab
    model_vocabs.dictVocab = dict_vocab

    factory = TSimpleJSONProtocol.TSimpleJSONProtocolFactory()
    json_str = Serializer.serialize(factory, model_vocabs)
    with open(out_file_path, "bw") as f:
        LOGGER.info("Writing serialized vocab to {}".format(out_file_path))
        f.write(json_str)


if __name__ == "__main__":
    args = PARSER.parse_args()
    token_vocab = load_vocab(args.intokenvocabfile)
    dict_vocab = load_vocab(args.indictvocabfile)
    serialize_model_vocabs(token_vocab, dict_vocab, args.outfile)
    LOGGER.info("Done")
