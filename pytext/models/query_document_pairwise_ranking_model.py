#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import os
from typing import Dict, List, Union

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.config.query_document_pairwise_ranking import ModelInput, ModelInputConfig
from pytext.data import CommonMetadata
from pytext.data.tensorizers import Tensorizer, TokenTensorizer
from pytext.models.decoders.mlp_decoder_query_response import MLPDecoderQueryResponse
from pytext.models.embeddings import EmbeddingBase, EmbeddingList, WordEmbedding
from pytext.models.model import Model
from pytext.models.module import create_module
from pytext.models.output_layers import PairwiseRankingOutputLayer
from pytext.models.pair_classification_model import BasePairwiseClassificationModel
from pytext.models.representations.bilstm_doc_attention import BiLSTMDocAttention
from pytext.models.representations.docnn import DocNNRepresentation
from pytext.models.representations.query_document_pairwise_ranking_rep import (
    QueryDocumentPairwiseRankingRep,
)


class QueryDocumentPairwiseRankingModel(Model):
    """Pairwise ranking model
    This model takes in a query, and two responses (pos_response and neg_response)
    It passes representations of the query and the two responses to a decoder
    pos_response should be ranked higher than neg_response - this is ensured by training
    with a ranking hinge loss function
    """

    class Config(ConfigBase):
        representation: QueryDocumentPairwiseRankingRep.Config = (
            QueryDocumentPairwiseRankingRep.Config()
        )
        decoder: MLPDecoderQueryResponse.Config = MLPDecoderQueryResponse.Config()
        output_layer: PairwiseRankingOutputLayer.Config = (
            PairwiseRankingOutputLayer.Config()
        )
        decoder_output_dim: int = 64

    @classmethod
    def from_config(cls, config, feat_config, metadata: CommonMetadata):
        embedding = create_module(
            feat_config, create_fn=cls.create_embedding, metadata=metadata
        )
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        # representation.representation_dim: tuple(2, actual repr dim)
        decoder = create_module(
            config.decoder,
            from_dim=representation.representation_dim[1],
            to_dim=config.decoder_output_dim,
        )
        output_layer = create_module(config.output_layer)
        return cls(embedding, representation, decoder, output_layer)

    @classmethod
    def create_sub_embs(
        cls, emb_config: ModelInputConfig, metadata: CommonMetadata
    ) -> Dict[str, EmbeddingBase]:
        """
        Creates the embedding modules defined in the `emb_config`.

        Args:
            emb_config (FeatureConfig): Object containing all the sub-embedding
                configurations.
            metadata (CommonMetadata): Object containing features and label metadata.

        Returns:
            Dict[str, EmbeddingBase]: Named dictionary of embedding modules.

        """
        sub_emb_module_dict = {}
        # create embeddings for response
        sub_emb_module_dict["response"] = create_module(
            emb_config.pos_response, metadata=metadata.features[ModelInput.POS_RESPONSE]
        )
        # TODO: allow query and response embeddings to be different
        return sub_emb_module_dict

    @classmethod
    def compose_embedding(cls, sub_embs, metadata):
        return EmbeddingList(sub_embs.values(), concat=False)

    def forward(self, *inputs) -> List[torch.Tensor]:
        embedding_input = inputs[:3]
        # inputs are: posResponse, negResponse, query
        # all three share the same embeddings
        token_emb = [
            self.embedding[0](embedding_input[0]),
            self.embedding[0](embedding_input[1]),
            self.embedding[0](embedding_input[2]),
        ]
        other_input = inputs[3:]
        input_representation = self.representation(token_emb, *other_input)
        if not isinstance(input_representation, (list, tuple)):
            input_representation = [input_representation]
        elif isinstance(input_representation[-1], tuple):
            # since some lstm based representations return states as (h0, c0)
            input_representation = input_representation[:-1]
        return self.decoder(
            *input_representation
        )  # returned Tensor's dim = (batch_size, num_classes)

    def save_modules(self, base_path: str = "", suffix: str = ""):
        super().save_modules(base_path, suffix)

        # Special case to also save the sub-representations separately, if needed.
        for i, subrep in enumerate(self.representation.subrepresentations):
            if getattr(subrep.config, "save_path", None):
                path = subrep.config.save_path + "-" + str(i) + suffix
                if base_path:
                    path = os.path.join(base_path, path)
                print(
                    f"Saving state of module {type(subrep).__name__} " f"to {path} ..."
                )
                torch.save(subrep.state_dict(), path)


class NewQueryDocumentPairwiseRankingModel(BasePairwiseClassificationModel):
    """Pairwise ranking model
    This model takes in a query, and two responses (pos_response and neg_response)
    It passes representations of the query and the two responses to a decoder
    pos_response should be ranked higher than neg_response - this is ensured by training
    with a ranking hinge loss function
    """

    class Config(BasePairwiseClassificationModel.Config):
        class ModelInput(BasePairwiseClassificationModel.Config.ModelInput):
            query: TokenTensorizer.Config = TokenTensorizer.Config(column="query")
            pos_response: TokenTensorizer.Config = TokenTensorizer.Config(
                column="pos_response"
            )
            neg_response: TokenTensorizer.Config = TokenTensorizer.Config(
                column="neg_response"
            )

        inputs: ModelInput = ModelInput()
        embedding: WordEmbedding.Config = WordEmbedding.Config()
        representation: Union[
            BiLSTMDocAttention.Config, DocNNRepresentation.Config
        ] = BiLSTMDocAttention.Config()
        # should the query and the response share representations?
        shared_representations: bool = True
        decoder: MLPDecoderQueryResponse.Config = MLPDecoderQueryResponse.Config()
        output_layer: PairwiseRankingOutputLayer.Config = (
            PairwiseRankingOutputLayer.Config()
        )
        decoder_output_dim: int = 64

    def __init__(
        self,
        embeddings: nn.ModuleList,
        representations: nn.ModuleList,
        decoder: MLPDecoderQueryResponse,
        output_layer: PairwiseRankingOutputLayer,
    ) -> None:
        super().__init__(representations, decoder, output_layer)
        self.embeddings = embeddings

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        embeddings = nn.ModuleList(
            [
                create_module(config.embedding, None, tensorizers[name])
                for name in ["query", "pos_response", "neg_response"]
            ]
        )
        embedding_dim = embeddings[0].embedding_dim

        if config.shared_representations:
            representations = nn.ModuleList(
                itertools.repeat(
                    create_module(config.representation, embed_dim=embedding_dim),
                    len(embeddings),
                )
            )
        else:
            # the first and second representations are run through the same LSTM
            representations = nn.ModuleList(
                itertools.repeat(
                    create_module(config.representation, embed_dim=embedding_dim), 2
                )
            )
            representations.append(
                create_module(config.representation, embed_dim=embedding_dim)
            )

        # representation.representation_dim: tuple(2, actual repr dim)
        decoder = create_module(
            config.decoder,
            from_dim=representations[0].representation_dim,
            to_dim=config.decoder_output_dim,
        )
        output_layer = create_module(config.output_layer)
        return cls(embeddings, representations, decoder, output_layer)

    def arrange_model_inputs(self, tensor_dict):
        qtokens, qseq_length = tensor_dict["query"]
        postokens, posseq_length = tensor_dict["pos_response"]
        negtokens, negseq_length = tensor_dict["neg_response"]
        return (
            [qtokens, postokens, negtokens],
            [qseq_length, posseq_length, negseq_length],
        )

    def arrange_targets(self, tensor_dict):
        return {}

    def forward(
        self, tokens: List[torch.Tensor], seq_lens: List[torch.Tensor]
    ) -> torch.Tensor:
        embeddings = [emb(token) for emb, token in zip(self.embeddings, tokens)]
        representations = self._represent(embeddings, seq_lens, self.representations)

        input_representation = []
        for representation in representations:
            if isinstance(representation, (list, tuple)):
                input_representation.append(representation[0])
            else:
                input_representation.append(representation)
        return self.decoder(*input_representation)
