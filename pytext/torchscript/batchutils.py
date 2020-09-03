#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional, Tuple


class pytextBatchSortClassEmbeddingModule:
    def __init__(
        self,
        myTuple: Tuple[
            Optional[List[str]],  # texts
            Optional[List[List[str]]],  # multi_texts
            Optional[List[List[str]]],  # tokens
            Optional[List[str]],  # language,
            Optional[List[List[float]]],  # dense_feat
            int,
        ],
        argno: int = 0,
    ):
        self.myTuple = myTuple
        self.argno = argno

    def __lt__(self, other: "pytextBatchSortClassEmbeddingModule") -> bool:

        argno = self.argno

        if argno == 0:
            this = self.myTuple[0]
            that = other.myTuple[0]

            if this is None:
                return True
            elif that is None:
                return False
            else:
                # TBD: tokenize this and that,
                # then sort by tokens instead of string length
                return max([len(x) for x in this]) < max([len(x) for x in that])
        else:
            raise NotImplementedError

        return False

    def t(
        self,
    ) -> Tuple[
        Optional[List[str]],  # texts
        Optional[List[List[str]]],  # multi_texts
        Optional[List[List[str]]],  # tokens
        Optional[List[str]],  # language,
        Optional[List[List[float]]],  # dense_feat
        int,
    ]:
        return self.myTuple


class pytextBatchSortClass:
    def __init__(
        self,
        myTuple: Tuple[
            Optional[List[str]],  # texts
            Optional[List[List[str]]],  # multi_texts
            Optional[List[List[str]]],  # tokens
            Optional[List[str]],  # language,
            int,
        ],
        argno: int = 0,
    ):
        self.myTuple = myTuple
        self.argno = argno

    def __lt__(self, other) -> bool:

        argno = self.argno

        if argno == 0:
            this = self.myTuple[0]
            that = other.myTuple[0]

            if this is None:
                return True
            elif that is None:
                return False
            else:
                return max([len(x) for x in this]) < max([len(x) for x in that])
        else:
            raise NotImplementedError

        return False

    def t(
        self,
    ) -> Tuple[
        Optional[List[str]],  # texts
        Optional[List[List[str]]],  # multi_texts
        Optional[List[List[str]]],  # tokens
        Optional[List[str]],  # language,
        int,
    ]:
        return self.myTuple
