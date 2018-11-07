#!/usr/bin/env python3


from pytext.models.module import Module


class EmbeddingBase(Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # By default has 1 embedding which is itself, for EmbeddingList, this num
        # can be greater than 1
        self.num_emb = 1
        self.embedding_dim = embedding_dim
