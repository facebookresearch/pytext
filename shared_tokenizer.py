#!/usr/bin/env python3
import aml.text.tokenizer_config.ttypes as tokenizer_config_types
from libfb.py.thrift_clients.configerator_thrift_client import ConfigeratorThriftClient
from admarket.hive_tools.modules.tokenizer import Tokenizer
from typing import List, Tuple
from pytext.common.constants import ConfigeratorPath


class SharedTokenizer(object):
    def __init__(self, config_path=ConfigeratorPath.DEFAULT_TOKENIZER):
        tokenizer_config = SharedTokenizer.get_tokenizer_config(config_path)
        self.tokenizer = Tokenizer(tokenizer_config)

    def tokenize(self, input: str) -> List[str]:
        return self.tokenizer.tokenize(input)

    def tokenize_with_ranges(self, input: str) -> List[Tuple[str, Tuple[int, int]]]:
        return self.tokenizer.tokenize_with_ranges(input)

    @staticmethod
    def get_tokenizer_config(config_path=ConfigeratorPath.DEFAULT_TOKENIZER):
        with ConfigeratorThriftClient() as ctc:
            tokenizer_config = ctc.getThriftConfigContent(
                config_path, tokenizer_config_types.TokenizerConfig
            )
            return tokenizer_config
