#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional, Tuple


class PytextEmbeddingModuleBatchSort:
    def __init__(
        self,
        batchElement: Tuple[
            Optional[List[str]],  # texts
            Optional[List[List[str]]],  # multi_texts
            Optional[List[List[str]]],  # tokens
            Optional[List[str]],  # language,
            Optional[List[List[float]]],  # dense_feat
            int,
        ],
        argno: int = 0,
    ):
        self.batchElement = batchElement
        self.argno = argno

    def __lt__(self, other: "PytextEmbeddingModuleBatchSort") -> bool:

        argno = self.argno

        if argno == 0:
            this = self.batchElement[0]
            that = other.batchElement[0]

            if this is None:
                return True
            elif that is None:
                return False
            else:
                # TBD: tokenize this and that,
                # then sort by tokens instead of string length
                # IGNORE LINTER message:
                # torch.jit.frontend.UnsupportedNodeError: GeneratorExp aren't supported
                return max([len(x) for x in this]) < max([len(x) for x in that])  # noqa
        else:
            raise NotImplementedError

        return False

    def be(
        self,
    ) -> Tuple[
        Optional[List[str]],  # texts
        Optional[List[List[str]]],  # multi_texts
        Optional[List[List[str]]],  # tokens
        Optional[List[str]],  # language,
        Optional[List[List[float]]],  # dense_feat
        int,
    ]:
        """return the batch element stored in this wrapper class"""
        return self.batchElement


class PytextTwoTowerEmbeddingModuleBatchSort:
    def __init__(
        self,
        batchElement: Tuple[
            Optional[List[str]],  # right_texts
            Optional[List[str]],  # left_texts
            Optional[List[List[str]]],  # right_tokens
            Optional[List[List[str]]],  # left_tokens
            Optional[List[str]],  # languages
            Optional[List[List[float]]],  # right_dense_feat
            Optional[List[List[float]]],  # left_dense_feat
            int,
        ],
        argno: int = 0,
    ):
        self.batchElement = batchElement
        self.argno = argno

    def __lt__(self, other: "PytextTwoTowerEmbeddingModuleBatchSort") -> bool:

        argno = self.argno

        if argno == 0:
            this = self.batchElement[0]
            that = other.batchElement[0]

            if this is None:
                return True
            elif that is None:
                return False
            else:
                # TBD: tokenize this and that,
                # then sort by tokens instead of string length
                # IGNORE LINTER message:
                # torch.jit.frontend.UnsupportedNodeError: GeneratorExp aren't supported
                return max([len(x) for x in this]) < max([len(x) for x in that])  # noqa
        else:
            raise NotImplementedError

        return False

    def be(
        self,
    ) -> Tuple[
        Optional[List[str]],  # right_texts
        Optional[List[str]],  # left_texts
        Optional[List[List[str]]],  # right_tokens
        Optional[List[List[str]]],  # left_tokens
        Optional[List[str]],  # languages
        Optional[List[List[float]]],  # right_dense_feat
        Optional[List[List[float]]],  # left_dense_feat
        int,
    ]:
        """return the batch element stored in this wrapper class"""
        return self.batchElement
