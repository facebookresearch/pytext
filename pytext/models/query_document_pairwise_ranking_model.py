#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.config.query_document_pairwise_ranking import ModelInput, ModelInputConfig
from pytext.data import CommonMetadata
from pytext.data.tensorizers import Tensorizer, TokenTensorizer, VocabBuilder
from pytext.models.decoders.mlp_decoder_query_response import MLPDecoderQueryResponse
from pytext.models.embeddings import EmbeddingBase, EmbeddingList
from pytext.models.model import Model
from pytext.models.module import create_module
from pytext.models.output_layers import PairwiseRankingOutputLayer
from pytext.models.pair_classification_model import PairwiseModel
from pytext.models.representations.query_document_pairwise_ranking_rep import (
    QueryDocumentPairwiseRankingRep,
)


class QueryDocumentPairwiseRankingModel_Deprecated(Model):
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


class QueryDocPairwiseRankingModel(PairwiseModel):
    """Pairwise ranking model
    This model takes in a query, and two responses (pos_response and neg_response)
    It passes representations of the query and the two responses to a decoder
    pos_response should be ranked higher than neg_response - this is ensured by training
    with a ranking hinge loss function
    """

    class Config(PairwiseModel.Config):
        class ModelInput(Model.Config.ModelInput):
            pos_response: TokenTensorizer.Config = TokenTensorizer.Config(
                column="pos_response"
            )
            neg_response: TokenTensorizer.Config = TokenTensorizer.Config(
                column="neg_response"
            )
            query: TokenTensorizer.Config = TokenTensorizer.Config(column="query")

        inputs: ModelInput = ModelInput()
        decoder: MLPDecoderQueryResponse.Config = MLPDecoderQueryResponse.Config()
        output_layer: PairwiseRankingOutputLayer.Config = (
            PairwiseRankingOutputLayer.Config()
        )
        decoder_output_dim: int = 64

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        # merge tensorizer vocab
        vocab_builder = VocabBuilder()
        for tensorizer in tensorizers.values():
            vocab_builder.add_all(tensorizer.vocab.counts)
        merged_vocab = vocab_builder.make_vocab()
        for tensorizer in tensorizers.values():
            tensorizer.vocab = merged_vocab

        # create embeddings
        positive_emb = create_module(
            config.embedding, None, tensorizers["pos_response"]
        )
        negative_emb = positive_emb
        query_emb = positive_emb
        embeddings = nn.ModuleList([positive_emb, negative_emb, query_emb])
        embedding_dim = embeddings[0].embedding_dim

        # create representations
        positive_repr = create_module(config.representation, embed_dim=embedding_dim)
        negative_repr = positive_repr
        query_repr = (
            positive_repr
            if config.shared_representations
            else create_module(config.representation, embed_dim=embedding_dim)
        )
        representations = nn.ModuleList([positive_repr, negative_repr, query_repr])

        # representation.representation_dim: tuple(2, actual repr dim)
        decoder = create_module(
            config.decoder,
            from_dim=representations[0].representation_dim,
            to_dim=config.decoder_output_dim,
        )
        output_layer = create_module(config.output_layer)
        return cls(embeddings, representations, decoder, output_layer, False)

    def arrange_model_inputs(self, tensor_dict):
        return (
            tensor_dict["pos_response"][:2],
            tensor_dict["neg_response"][:2],
            tensor_dict["query"][:2],
        )

    def arrange_targets(self, tensor_dict):
        return {}

    def forward(
        self,
        pos_response: Tuple[torch.Tensor, torch.Tensor],
        neg_response: Tuple[torch.Tensor, torch.Tensor],
        query: Tuple[torch.Tensor, torch.Tensor],
    ) -> List[torch.Tensor]:
        tokens, seq_lens = list(zip(pos_response, neg_response, query))
        embeddings = [emb(token) for emb, token in zip(self.embeddings, tokens)]
        representations = self._represent_sort(
            embeddings, seq_lens, self.representations
        )
        return self.decoder(*representations)
