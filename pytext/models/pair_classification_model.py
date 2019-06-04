#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import os
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from pytext.data.tensorizers import LabelTensorizer, Tensorizer, TokenTensorizer
from pytext.models.decoders import DecoderBase
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.embeddings import EmbeddingBase, EmbeddingList, WordEmbedding
from pytext.models.model import BaseModel
from pytext.models.module import create_module
from pytext.models.output_layers import ClassificationOutputLayer, OutputLayerBase
from pytext.models.representations.bilstm_doc_attention import BiLSTMDocAttention
from pytext.models.representations.docnn import DocNNRepresentation
from pytext.models.representations.representation_base import RepresentationBase
from scipy.special import comb


class BasePairwiseModel(BaseModel):
    """
    A base classification model that scores a pair of texts.

    Subclasses need to implement the from_config, forward and save_modules.
    """

    __EXPANSIBLE__ = True

    class Config(BaseModel.Config):
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )
        encode_relations: bool = True

    def __init__(
        self,
        decoder: DecoderBase,
        output_layer: OutputLayerBase,
        encode_relations: bool,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.output_layer = output_layer
        self.encode_relations = encode_relations

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        raise NotImplementedError

    def forward(
        self, input1: Tuple[torch.Tensor, ...], input2: Tuple[torch.Tensor, ...]
    ):
        raise NotImplementedError

    def save_modules(self, base_path: str = "", suffix: str = ""):
        raise NotImplementedError

    @classmethod
    def _create_decoder(
        cls,
        config: Config,
        representations: nn.ModuleList,
        tensorizers: Dict[str, Tensorizer],
    ):
        labels = tensorizers["labels"].vocab
        num_reps = len(representations)
        rep_dim = representations[0].representation_dim
        decoder_in_dim = num_reps * rep_dim
        if config.encode_relations:
            decoder_in_dim += 2 * comb(num_reps, 2, exact=True) * rep_dim

        decoder = create_module(
            config.decoder, in_dim=decoder_in_dim, out_dim=len(labels)
        )
        output_layer = create_module(config.output_layer, labels=labels)
        return decoder, output_layer

    @classmethod
    def _encode_relations(cls, encodings: List[torch.Tensor]) -> List[torch.Tensor]:
        for rep_l, rep_r in itertools.combinations(encodings, 2):
            encodings.append(torch.abs(rep_l - rep_r))
            encodings.append(rep_l * rep_r)
        return encodings

    def _save_modules(self, modules: nn.ModuleList, base_path: str, suffix: str):
        super().save_modules(base_path, suffix)
        # Special case to also save the multi-representations separately, if needed.
        for module in modules:
            if getattr(module.config, "save_path", None):
                path = module.config.save_path + suffix
                if base_path:
                    path = os.path.join(base_path, path)
                print(
                    f"Saving state of module {type(module).__name__} " f"to {path} ..."
                )
                torch.save(module.state_dict(), path)


class PairwiseModel(BasePairwiseModel):
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

    EMBEDDINGS = ["embedding"]
    INPUTS_PAIR = [["tokens1"], ["tokens2"]]

    class Config(BasePairwiseModel.Config):
        """
        Attributes:
            encode_relations (bool): if `false`, return the concatenation of the two
                representations; if `true`, also concatenate their pairwise absolute
                difference and pairwise elementwise product (à la arXiv:1705.02364).
                Default: `true`.
            tied_representation: whether to use the same representation, with
              tied weights, for all the input subrepresentations. Default: `true`.
        """

        class ModelInput(BasePairwiseModel.Config.ModelInput):
            tokens1: TokenTensorizer.Config = TokenTensorizer.Config(column="text1")
            tokens2: TokenTensorizer.Config = TokenTensorizer.Config(column="text2")
            labels: LabelTensorizer.Config = LabelTensorizer.Config()

        inputs: ModelInput = ModelInput()
        embedding: WordEmbedding.Config = WordEmbedding.Config()
        representation: Union[
            BiLSTMDocAttention.Config, DocNNRepresentation.Config
        ] = BiLSTMDocAttention.Config()
        shared_representations: bool = True

    def __init__(
        self,
        embeddings: nn.ModuleList,
        representations: nn.ModuleList,
        decoder: MLPDecoder,
        output_layer: ClassificationOutputLayer,
        encode_relations: bool,
    ) -> None:
        super().__init__(decoder, output_layer, encode_relations)
        self.embeddings = embeddings
        self.representations = representations

    # from_config and helper function
    @classmethod
    def _create_embedding(cls, config, tensorizer) -> EmbeddingBase:
        return create_module(config, None, tensorizer)

    @classmethod
    def _create_embeddings(
        cls, config: Config, tensorizers: Dict[str, Tensorizer]
    ) -> nn.ModuleList:
        embeddings = []
        for inputs in cls.INPUTS_PAIR:
            embedding_list = []
            for emb, input in zip(cls.EMBEDDINGS, inputs):
                if hasattr(config, emb) and input in tensorizers:
                    embedding_list.append(
                        cls._create_embedding(getattr(config, emb), tensorizers[input])
                    )

            if len(embedding_list) == 1:
                embeddings.append(embedding_list[0])
            else:
                embeddings.append(EmbeddingList(embeddings=embedding_list, concat=True))
        return nn.ModuleList(embeddings)

    @classmethod
    def _create_representations(cls, config: Config, embeddings: nn.ModuleList):
        if config.shared_representations:
            # create representation once and used for all embeddings
            embedding_dim = embeddings[0].embedding_dim
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
        return representations

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        embeddings = cls._create_embeddings(config, tensorizers)
        representations = cls._create_representations(config, embeddings)
        decoder, output_layer = cls._create_decoder(
            config, representations, tensorizers
        )
        return cls(
            embeddings, representations, decoder, output_layer, config.encode_relations
        )

    def arrange_model_inputs(self, tensor_dict):
        return tensor_dict["tokens1"][:2], tensor_dict["tokens2"][:2]

    def arrange_targets(self, tensor_dict):
        return tensor_dict["labels"]

    # _encode and helper functions
    @classmethod
    def _represent_helper(
        cls, rep: RepresentationBase, embs: torch.Tensor, lens: torch.Tensor
    ) -> torch.Tensor:
        representation = rep(embs, lens)
        if isinstance(representation, tuple):
            return representation[0]
        return representation

    @classmethod
    def _represent_sort(
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
        if isinstance(represention_modules[0], BiLSTMDocAttention):
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
        else:
            return [
                cls._represent_helper(rep, embs, lens)
                for rep, (embs, lens) in zip(
                    represention_modules, zip(embeddings, lengths)
                )
            ]

    def _represent(self, embeddings: List[torch.Tensor], seq_lens: List[torch.Tensor]):
        representations = self._represent_sort(
            embeddings, seq_lens, self.representations
        )
        if self.encode_relations:
            representations = self._encode_relations(representations)
        return torch.cat(representations, -1)

    def forward(
        self, input1: Tuple[torch.Tensor, ...], input2: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        token_tups, seq_lens = (input1[:-1], input2[:-1]), (input1[-1], input2[-1])
        embeddings = [
            emb(*token_tup) for emb, token_tup in zip(self.embeddings, token_tups)
        ]
        representation = self._represent(embeddings, seq_lens)
        return self.decoder(representation)

    def save_modules(self, base_path: str = "", suffix: str = ""):
        self._save_modules(self.representations, base_path, suffix)
