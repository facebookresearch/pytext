#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import os
from typing import Dict, List, Union

import torch
import torch.nn as nn
from pytext.config import ConfigBase
from pytext.data.tensorizers import (
    JoinStringTensorizer,
    LabelTensorizer,
    Tensorizer,
    TokenTensorizer,
)
from pytext.models.decoders import DecoderBase
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.embeddings import EmbeddingList, WordEmbedding
from pytext.models.model import BaseModel, Model
from pytext.models.module import create_module
from pytext.models.output_layers import ClassificationOutputLayer, OutputLayerBase
from pytext.models.representations.bilstm_doc_attention import BiLSTMDocAttention
from pytext.models.representations.docnn import DocNNRepresentation
from pytext.models.representations.pair_rep import PairRepresentation
from pytext.models.representations.representation_base import RepresentationBase
from scipy.special import comb


class PairClassificationModel(Model):
    """
    A classification model that scores a pair of texts, for example, a model for
    natural language inference.

    The model shares embedding space (so it doesn't support
    pairs of texts where left and right are in different languages). It uses
    bidirectional LSTM or CNN to represent the two documents, and concatenates
    them along with their absolute difference and elementwise product. This
    concatenated pair representation is passed to a multi-layer perceptron to
    decode to label/target space.

    See https://arxiv.org/pdf/1705.02364.pdf for more details.

    It can be instantiated just like any other :class:`~Model`.
    """

    class Config(ConfigBase):
        representation: PairRepresentation.Config = PairRepresentation.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        # TODO: will need to support different output layer for contrastive loss
        output_layer: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )

    @classmethod
    def compose_embedding(cls, sub_embs, metadata):
        return EmbeddingList(sub_embs.values(), concat=False)

    def save_modules(self, base_path: str = "", suffix: str = ""):
        super().save_modules(base_path, suffix)

        # Special case to also save the sub-representations separately, if needed.
        for subrep in self.representation.subrepresentations:
            if getattr(subrep.config, "save_path", None):
                path = subrep.config.save_path + suffix
                if base_path:
                    path = os.path.join(base_path, path)
                print(
                    f"Saving state of module {type(subrep).__name__} " f"to {path} ..."
                )
                torch.save(subrep.state_dict(), path)


class BasePairwiseClassificationModel(BaseModel):
    def __init__(
        self,
        representations: nn.ModuleList,
        decoder: DecoderBase,
        output_layer: OutputLayerBase,
    ) -> None:
        super().__init__()
        self.representations = representations
        self.decoder = decoder
        self.output_layer = output_layer

    @staticmethod
    def _representation_dim(representations, encode_relations):
        num_reps = len(representations)
        rep_dim = representations[0].representation_dim

        representation_dim = num_reps * rep_dim
        if encode_relations:
            representation_dim += 2 * comb(num_reps, 2, exact=True) * rep_dim
        return representation_dim

    @staticmethod
    def _represent_helper(
        rep: RepresentationBase, embs: torch.Tensor, lens: torch.Tensor
    ) -> torch.Tensor:
        representation = rep(embs, lens)
        if isinstance(representation, tuple):
            return representation[0]
        return representation

    @classmethod
    def _represent(
        cls,
        embeddings: List[torch.Tensor],
        lengths: List[torch.Tensor],
        represention_modules: nn.ModuleList,
    ) -> List[torch.Tensor]:
        """
        Apply the representations computations in `self.representations` to the
        sentence representations in `embeddings`.
        Internally, it sorts the sentences in `embeddings` by the number
        of tokens for packing efficiency, where the number of tokens is in `lengths`,
        and undoes the sort after applying the representations to preserve the
        original ordering of sentences. Assumes that the leftmost sentences are
        already sorted by number of tokens.
        """
        # The leftmost inputs already come sorted by length. The others need to
        # be sorted as well, for packing. We do it manually.
        sorted_inputs = [(embeddings[0], lengths[0])]
        sorted_indices = [None]
        for embs, lens in zip(embeddings[1:], lengths[1:]):
            lens_sorted, sorted_idx = lens.sort(descending=True)
            embs_sorted = embs[sorted_idx]
            sorted_inputs.append((embs_sorted, lens_sorted))
            sorted_indices.append(sorted_idx)

        representations = [
            cls._represent_helper(rep, embs, lens)
            for rep, (embs, lens) in zip(represention_modules, sorted_inputs)
        ]

        # Put the inputs back in the original order, so they still match up to
        # each other as well as the targets.
        unsorted_representations = [representations[0]]
        for sorted_idx, rep in zip(sorted_indices[1:], representations[1:]):
            _, unsorted_idx = sorted_idx.sort()
            unsorted_representations.append(rep[unsorted_idx])
        return unsorted_representations

    def _represent_encode_relations(
        self, representations: List[torch.Tensor]
    ) -> torch.Tensor:
        for rep_l, rep_r in itertools.combinations(representations, 2):
            representations.append(torch.abs(rep_l - rep_r))
            representations.append(rep_l * rep_r)
        return representations

    def save_modules(self, base_path: str = "", suffix: str = ""):
        super().save_modules(base_path, suffix)

        # Special case to also save the multi-representations separately, if needed.
        for representation in self.representations:
            if getattr(representation.config, "save_path", None):
                path = representation.config.save_path + suffix
                if base_path:
                    path = os.path.join(base_path, path)
                print(
                    f"Saving state of module {type(representation).__name__} "
                    f"to {path} ..."
                )
                torch.save(representation.state_dict(), path)


class PairwiseClassificationModel(BasePairwiseClassificationModel):
    """
    A classification model that scores a pair of texts, for example, a model for
    natural language inference.

    The model shares embedding space (so it doesn't support
    pairs of texts where left and right are in different languages). It uses
    bidirectional LSTM or CNN to represent the two documents, and concatenates
    them along with their absolute difference and elementwise product. This
    concatenated pair representation is passed to a multi-layer perceptron to
    decode to label/target space.

    See https://arxiv.org/pdf/1705.02364.pdf for more details.

    It can be instantiated just like any other :class:`~Model`.
    """

    class Config(BasePairwiseClassificationModel.Config):
        """
        Attributes:
            encode_relations (bool): if `false`, return the concatenation of the two
                representations; if `true`, also concatenate their pairwise absolute
                difference and pairwise elementwise product (à la arXiv:1705.02364).
                Default: `true`.
            tied_representation: whether to use the same representation, with
              tied weights, for all the input subrepresentations. Default: `true`.
        """

        class ModelInput(BasePairwiseClassificationModel.Config.ModelInput):
            tokens1: TokenTensorizer.Config = TokenTensorizer.Config(column="text1")
            tokens2: TokenTensorizer.Config = TokenTensorizer.Config(column="text2")
            labels: LabelTensorizer.Config = LabelTensorizer.Config()
            # for metric reporter
            raw_text: JoinStringTensorizer.Config = JoinStringTensorizer.Config(
                columns=["text1", "text2"]
            )

        inputs: ModelInput = ModelInput()
        embedding: WordEmbedding.Config = WordEmbedding.Config()
        representation: Union[
            BiLSTMDocAttention.Config, DocNNRepresentation.Config
        ] = BiLSTMDocAttention.Config()
        shared_representations: bool = True
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        # TODO: will need to support different output layer for contrastive loss
        output_layer: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )
        encode_relations: bool = True

    def __init__(
        self,
        embeddings: nn.ModuleList,
        representations: nn.ModuleList,
        decoder: MLPDecoder,
        output_layer: ClassificationOutputLayer,
        encode_relations: bool,
    ) -> None:
        super().__init__(representations, decoder, output_layer)
        self.embeddings = embeddings
        self.encode_relations = encode_relations

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        labels = tensorizers["labels"].labels

        # len(embeddings) == 2
        embeddings = nn.ModuleList(
            [
                create_module(config.embedding, None, tensorizers[name])
                for name in ["tokens1", "tokens2"]
            ]
        )
        embedding_dim = embeddings[0].embedding_dim

        if config.shared_representations:
            # create representation once and used for all embeddings
            representations = nn.ModuleList(
                itertools.repeat(
                    create_module(config.representation, embed_dim=embedding_dim),
                    len(embeddings),
                )
            )
        else:
            representations = nn.ModuleList(
                [
                    create_module(
                        config.representation, embed_dim=embedding.embedding_dim
                    )
                    for embedding in embeddings
                ]
            )

        decoder_in_dim = cls._representation_dim(
            representations, config.encode_relations
        )
        decoder = create_module(
            config.decoder, in_dim=decoder_in_dim, out_dim=len(labels)
        )
        output_layer = create_module(config.output_layer, labels=labels)
        return cls(
            embeddings, representations, decoder, output_layer, config.encode_relations
        )

    def arrange_model_inputs(self, tensor_dict):
        tokens1, seq_length1 = tensor_dict["tokens1"]
        tokens2, seq_length2 = tensor_dict["tokens2"]
        return [tokens1, tokens2], [seq_length1, seq_length2]

    def arrange_targets(self, tensor_dict):
        return tensor_dict["labels"]

    def forward(
        self, tokens: List[torch.Tensor], seq_lens: List[torch.Tensor]
    ) -> torch.Tensor:
        embeddings = [emb(token) for emb, token in zip(self.embeddings, tokens)]
        representations = self._represent(embeddings, seq_lens, self.representations)
        if self.encode_relations:
            representations = self._represent_encode_relations(representations)
        representation = torch.cat(representations, -1)

        return self.decoder(representation)
