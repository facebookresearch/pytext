#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from pytext.common.constants import Stage
from pytext.config.component import Component, ComponentType, create_loss
from pytext.config.doc_classification import ModelInput
from pytext.config.field_config import FeatureConfig
from pytext.config.pytext_config import ConfigBase, ConfigBaseMeta
from pytext.config.serialize import _is_optional
from pytext.data import CommonMetadata
from pytext.data.tensorizers import (
    CharacterTokenTensorizer,
    DictTensorizer,
    FloatListTensorizer,
    LabelTensorizer,
    RawString,
    Tensorizer,
    TokenTensorizer,
)
from pytext.data.utils import PAD, UNK
from pytext.exporters.exporter import ModelExporter
from pytext.loss import BinaryCrossEntropyLoss
from pytext.models.module import create_module
from pytext.models.output_layers.doc_classification_output_layer import (
    BinaryClassificationOutputLayer,
    MulticlassOutputLayer,
)
from pytext.utils.precision import maybe_float
from pytext.utils.torch import Vocabulary, list_max
from torch import jit

from .decoders import DecoderBase
from .embeddings import (
    CharacterEmbedding,
    DictEmbedding,
    EmbeddingBase,
    EmbeddingList,
    WordEmbedding,
)
from .output_layers import OutputLayerBase
from .representations.representation_base import RepresentationBase


def _assert_tensorizer_type(t):
    if t is not type(None) and not issubclass(t, Tensorizer.Config):
        raise TypeError(
            f"ModelInput configuration should only include tensorizers: {t}"
        )


class ModelInputMeta(ConfigBaseMeta):
    def __new__(metacls, typename, bases, namespace):
        annotations = namespace.get("__annotations__", {})
        for t in annotations.values():
            if type(t) == type(Union):
                for ut in t.__args__:
                    _assert_tensorizer_type(ut)
            else:
                _assert_tensorizer_type(t)
        return super().__new__(metacls, typename, bases, namespace)


class ModelInputBase(ConfigBase, metaclass=ModelInputMeta):
    """Base class for model inputs."""


class BaseModel(nn.Module, Component):
    """
    Base model class which inherits from nn.Module. Also has a stage flag to
    indicate it's in `train`, `eval`, or `test` stage.
    This is because the built-in train/eval flag in PyTorch can't distinguish eval
    and test, which is required to support some use cases.
    """

    __EXPANSIBLE__ = True
    __COMPONENT_TYPE__ = ComponentType.MODEL

    SUPPORT_FP16_OPTIMIZER = False

    class Config(Component.Config):
        class ModelInput(ModelInputBase):
            pass

        inputs: ModelInput = ModelInput()

    def __init__(self, stage: Stage = Stage.TRAIN) -> None:
        nn.Module.__init__(self)
        self.stage = stage
        self.module_list: List[nn.Module] = []
        self.find_unused_parameters = True

    def train(self, mode=True):
        """Override to explicitly maintain the stage (train, eval, test)."""
        super().train(mode)
        self.stage = Stage.TRAIN

    def eval(self, stage=Stage.TEST):
        """Override to explicitly maintain the stage (train, eval, test)."""
        super().eval()
        self.stage = stage

    def contextualize(self, context):
        """Add additional context into model. `context` can be anything that
        helps maintaining/updating state. For example, it is used by
        :class:`~DisjointMultitaskModel` for changing the task that should be
        trained with a given iterator.
        """
        self.context = context

    def get_loss(self, logit, target, context):
        return maybe_float(self.output_layer.get_loss(logit, target, context))

    def get_pred(self, logit, target=None, context=None, *args):
        return self.output_layer.get_pred(logit, target, context)

    def save_modules(self, base_path: str = "", suffix: str = ""):
        """Save each sub-module in separate files for reusing later."""
        for module in self.module_list:
            if module and getattr(module.config, "save_path", None):
                path = module.config.save_path + suffix
                if base_path:
                    path = os.path.join(base_path, path)
                print(f"Saving state of module {type(module).__name__} to {path} ...")
                torch.save(module.state_dict(), path)

    def prepare_for_onnx_export_(self, **kwargs):
        """Make model exportable via ONNX trace."""

        def apply_prepare_for_onnx_export_(module):
            if module != self and hasattr(module, "prepare_for_onnx_export_"):
                module.prepare_for_onnx_export_(**kwargs)

        self.apply(apply_prepare_for_onnx_export_)

    def get_param_groups_for_optimizer(self) -> List[Dict[str, List[nn.Parameter]]]:
        """
        Returns a list of parameter groups of the format {"params": param_list}.
        The parameter groups loosely correspond to layers and are ordered from low
        to high. Currently, only the embedding layer can provide multiple param groups,
        and other layers are put into one param group. The output of this method
        is passed to the optimizer so that schedulers can change learning rates
        by layer.
        """
        non_emb_params = dict(self.named_parameters())
        model_params = [non_emb_params]

        # some subclasses of Model (e.g. Ensemble) do not have embeddings
        embedding = getattr(self, "embedding", None)
        if embedding is not None:
            emb_params_by_layer = self.embedding.get_param_groups_for_optimizer()

            # Delete params from the embedding layers
            for emb_params in emb_params_by_layer:
                for name in emb_params:
                    del non_emb_params["embedding.%s" % name]

            model_params = emb_params_by_layer + model_params
            print_str = (
                "Model has %d param groups (%d from embedding module) for optimizer"
            )
            print(print_str % (len(model_params), len(emb_params_by_layer)))

        model_params = [{"params": params.values()} for params in model_params]
        return model_params

    ##################################
    #    New Model functions         #
    ##################################
    # TODO: add back after migration
    # @classmethod
    # def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
    #     raise NotImplementedError
    @classmethod
    def train_batch(cls, model, batch):
        # This is a class method so that it works when model is a DistributedModel
        # wrapper. Otherwise the forward call here skips the DDP forward call.

        # Forward pass through the network.
        model_inputs = model.arrange_model_inputs(batch)
        targets = model.arrange_targets(batch)
        model_outputs = model(*model_inputs)

        # Compute loss and predictions.
        loss = model.get_loss(model_outputs, targets, None)
        predictions, scores = model.get_pred(model_outputs)

        # Pack results and return them.
        metric_data = (predictions, targets, scores, loss, model_inputs)
        return loss, metric_data

    def arrange_model_inputs(self, tensor_dict):
        # should raise NotImplementedError after migration is done
        pass

    def arrange_targets(self, tensor_dict):
        # should raise NotImplementedError after migration is done
        pass

    def caffe2_export(self, tensorizers, tensor_dict, path, export_onnx_path=None):
        pass


class Model(BaseModel):
    """
    Generic single-task model class that expects four components:

    1. `Embedding`
    2. `Representation`
    3. `Decoder`
    4. `Output Layer`

    Forward pass: `embedding -> representation -> decoder -> output_layer`

    These four components have specific responsibilities as described below.

    `Embedding` layer should implement the way to represent each token in the
    input text. It can be as simple as just token/word embedding or can be
    composed of multiple ways to represent a token, e.g., word embedding,
    character embedding, etc.

    `Representation` layer should implement the way to encode the entire input
    text such that the output vector(s) can be used by decoder to produce logits.
    There is no restriction on the number of inputs it should encode. There is
    also not restriction on the number of ways to encode input.

    `Decoder` layer should implement the way to consume the output of model's
    representation and produce logits that can be used by the output layer to
    compute loss or generate predictions (and prediction scores/confidence)

    `Output layer` should implement the way loss computation is done as well as
    the logic to generate predictions from the logits.

    Let us discuss the joint intent-slot model as a case to go over these layers.
    The model predicts intent of input utterance and the slots in the utterance.
    (Refer to :doc:`/atis_tutorial` for details about intent-slot model.)

    1. :class:`~EmbeddingList` layer is tasked with representing tokens. To do so we
       can use learnable word embedding table in conjunction with learnable character
       embedding table that are distilled to token level representation using CNN and
       pooling.
       Note: This class is meant to be reused by all models. It acts as a container
       of all the different ways of representing a token/word.
    2. :class:`~BiLSTMDocSlotAttention` is tasked with encoding the embedded input
       string for intent classification and slot filling. In order to do that it has a
       shared bidirectional LSTM layer followed by separate attention layers for
       document level attention and word level attention. Finally it produces two
       vectors per utterance.
    3. :class:`~IntentSlotModelDecoder` accepts the two input vectors from
       `BiLSTMDocSlotAttention` and produces logits for intent classification and
       slot filling. Conditioned on a flag it can also use the probabilities from
       intent classification for slot filling.
    4. :class:`~IntentSlotOutputLayer` implements the logic behind computing loss and
       prediction, as well as, how to export this layer to export to Caffe2. This is
       used by model exporter as a post-processing Caffe2 operator.


    Args:
        embedding (EmbeddingBase): Description of parameter `embedding`.
        representation (RepresentationBase): Description of parameter `representation`.
        decoder (DecoderBase): Description of parameter `decoder`.
        output_layer (OutputLayerBase): Description of parameter `output_layer`.

    Attributes:
        embedding
        representation
        decoder
        output_layer

    """

    __EXPANSIBLE__ = True

    class Config(BaseModel.Config):
        representation = None
        decoder = None
        output_layer = None

    def __init__(
        self,
        embedding: EmbeddingBase,
        representation: RepresentationBase,
        decoder: DecoderBase,
        output_layer: OutputLayerBase,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.representation = representation
        self.decoder = decoder
        self.output_layer = output_layer
        self.module_list = [embedding, representation, decoder]

    @classmethod
    def create_sub_embs(
        cls, emb_config: FeatureConfig, metadata: CommonMetadata
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
        for name, config in emb_config._asdict().items():
            if issubclass(getattr(config, "__COMPONENT__", object), EmbeddingBase):
                sub_emb_module_dict[name] = create_module(
                    config, metadata=metadata.features[name]
                )
            else:
                print(f"{name} is not a config of embedding, skipping")
        return sub_emb_module_dict

    @classmethod
    def compose_embedding(
        cls, sub_emb_module_dict: Dict[str, EmbeddingBase], metadata
    ) -> EmbeddingList:
        """Default implementation is to compose an instance of
        :class:`~EmbeddingList` with all the sub-embedding modules. You should
        override this class method if you want to implement a specific way to
        embed tokens/words.

        Args:
            sub_emb_module_dict (Dict[str, EmbeddingBase]): Named dictionary of
                embedding modules each of which implement a way to embed/encode
                a token.

        Returns:
            EmbeddingList: An instance of :class:`~EmbeddingList`.

        """
        return EmbeddingList(sub_emb_module_dict.values(), concat=True)

    @classmethod
    def create_embedding(cls, feat_config: FeatureConfig, metadata: CommonMetadata):
        sub_emb_module_dict = cls.create_sub_embs(feat_config, metadata)
        emb_module = cls.compose_embedding(sub_emb_module_dict, metadata)
        emb_module.config = feat_config
        return emb_module

    @classmethod
    def from_config(
        cls, config: Config, feat_config: FeatureConfig, metadata: CommonMetadata
    ):
        embedding = create_module(
            feat_config, create_fn=cls.create_embedding, metadata=metadata
        )
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        # Find all inputs for the decoder layer
        decoder_in_dim = representation.representation_dim
        decoder_input_features_count = 0
        for decoder_feat in (ModelInput.DENSE_FEAT,):  # Only 1 right now.
            if getattr(feat_config, decoder_feat, False):
                decoder_input_features_count += 1
                decoder_in_dim += getattr(feat_config, ModelInput.DENSE_FEAT).dim
        decoder = create_module(
            config.decoder, in_dim=decoder_in_dim, out_dim=metadata.target.vocab_size
        )
        decoder.num_decoder_modules = decoder_input_features_count
        output_layer = create_module(config.output_layer, metadata.target)
        return cls(embedding, representation, decoder, output_layer)

    def forward(self, *inputs) -> List[torch.Tensor]:
        embedding_input = inputs[: self.embedding.num_emb_modules]
        token_emb = self.embedding(*embedding_input)
        other_input = inputs[
            self.embedding.num_emb_modules : len(inputs)
            - self.decoder.num_decoder_modules
        ]
        input_representation = self.representation(token_emb, *other_input)
        if not isinstance(input_representation, (list, tuple)):
            input_representation = [input_representation]
        elif isinstance(input_representation[-1], tuple):
            # since some lstm based representations return states as (h0, c0)
            input_representation = input_representation[:-1]
        decoder_inputs: tuple = ()
        if self.decoder.num_decoder_modules:
            decoder_inputs = inputs[-self.decoder.num_decoder_modules :]
        return self.decoder(
            *input_representation, *decoder_inputs
        )  # returned Tensor's dim = (batch_size, num_classes)


class SimpleTokenModel(BaseModel):
    """Base Model that supports upto three token level inputs, and one document
    level (dense) feature. Token inputs are passed to their respective embeddings,
    concatenated and fed into the representation layer. Output of the
    representation layer is concatanated with doc features and fed into
    the decoder."""

    __EXPANSIBLE__ = True
    INPUT_NAMES = ["tokens", "characters", "dicts"]

    class Config(BaseModel.Config):
        class ModelInput(BaseModel.Config.ModelInput):
            # supports up to 3 token level inputs, e.g. TokenTensorizer,
            # CharTokenTensorizer or DictTensorizer
            tokens: Optional[TokenTensorizer.Config] = None
            characters: Optional[CharacterTokenTensorizer.Config] = None
            dicts: Optional[DictTensorizer.Config] = None

            # supports document level dense features
            dense: Optional[FloatListTensorizer.Config] = None
            # labels
            labels: LabelTensorizer.Config = LabelTensorizer.Config(allow_unknown=True)
            # for metric reporter
            raw_text: RawString.Config = RawString.Config(column="text")

        inputs: ModelInput = ModelInput()
        token_embedding: Optional[WordEmbedding.Config] = None
        character_embedding: Optional[CharacterEmbedding.Config] = None
        dict_embedding: Optional[DictEmbedding.Config] = None

        representation: RepresentationBase.Config
        decoder: DecoderBase.Config
        output_layer: OutputLayerBase.Config

    @classmethod
    def from_config(
        cls, config: Config, tensorizers: Dict[str, Tensorizer], metadata=None
    ):
        embedding_dim = 0

        character_embedding = None
        if config.character_embedding or "characters" in tensorizers:
            emb_config = config.character_embedding or CharacterEmbedding.Config()
            tsrz = tensorizers.get("characters", CharacterEmbedding.Config())
            character_embedding = create_module(emb_config, tensorizer=tsrz)
            embedding_dim += character_embedding.embedding_dim

        dict_embedding = None
        if config.dict_embedding or "dicts" in tensorizers:
            emb_config = config.dict_embedding or DictEmbedding.Config()
            tsrz = tensorizers.get("dicts", DictEmbedding.Config())
            dict_embedding = create_module(emb_config, tensorizer=tsrz)
            embedding_dim += dict_embedding.embedding_dim

        # If nothing configured so far, assume default tokens
        if config.token_embedding  or "tokens" in tensorizers or embedding_dim == 0:
            emb_config = config.token_embedding or WordEmbedding.Config()
            tsrz = tensorizers.get("tokens", WordEmbedding.Config())
            token_embedding = create_module(emb_config, tensorizer=tsrz)
            embedding_dim = token_embedding.embedding_dim

        representation = create_module(config.representation, embed_dim=embedding_dim)
        doc_feature_dim = tensorizers["dense"].out_dim if "dense" in tensorizers else 0

        labels = tensorizers["labels"].vocab
        decoder = create_module(
            config.decoder,
            in_dim=representation.representation_dim + doc_feature_dim,
            out_dim=len(labels),
        )

        loss = create_loss(config.output_layer.loss)
        output_layer_cls = (
            BinaryClassificationOutputLayer
            if isinstance(loss, BinaryCrossEntropyLoss)
            else MulticlassOutputLayer
        )
        output_layer = output_layer_cls(list(labels), loss)
        return cls(
            token_embedding=token_embedding,
            character_embedding=character_embedding,
            dict_embedding=dict_embedding,
            representation=representation,
            decoder=decoder,
            output_layer=output_layer,
        )

    def __init__(
        self,
        representation: RepresentationBase,
        decoder: DecoderBase,
        output_layer: OutputLayerBase,
        token_embedding: WordEmbedding,
        character_embedding: Optional[EmbeddingBase] = None,
        dict_embedding: Optional[DictEmbedding] = None,
    ) -> None:
        super().__init__()
        self.embedding = EmbeddingList(
            [
                emb
                for emb in (token_embedding, character_embedding, dict_embedding)
                if emb is not None
            ],
            concat=True,
        )
        self.representation = representation
        self.decoder = decoder
        self.output_layer = output_layer

        # needed by save_modules
        self.module_list = [
            token_embedding,
            character_embedding,
            dict_embedding,
            representation,
            decoder,
        ]

    def arrange_model_inputs(self, tensor_dict):
        res = tuple(
            tensor_dict[name] for name in self.INPUT_NAMES if name in tensor_dict
        )
        if "dense" in tensor_dict:
            res += (tensor_dict["dense"],)
        return res

    def arrange_targets(self, tensor_dict):
        return tensor_dict["labels"]

    def get_export_input_names(self, tensorizers):
        return [
            x
            for name in self.INPUT_NAMES
            for x in (name, name + "_lens")
            if name in tensorizers
        ]

    def get_export_output_names(self, tensorizers):
        return ["scores"]

    def vocab_to_export(self, tensorizers):
        return {
            name: list(tensorizers[name].vocab)
            for name in self.INPUT_NAMES
            if name in tensorizers
        }

    def caffe2_export(self, tensorizers, tensor_dict, path, export_onnx_path=None):
        exporter = ModelExporter(
            config=ModelExporter.Config(),
            input_names=self.get_export_input_names(tensorizers),
            dummy_model_input=self.arrange_model_inputs(tensor_dict),
            vocab_map=self.vocab_to_export(tensorizers),
            output_names=self.get_export_output_names(tensorizers),
        )
        return exporter.export_to_caffe2(self, path, export_onnx_path=export_onnx_path)

    def torchscriptify(self, tensorizers, traced_model):
        output_layer = self.output_layer.torchscript_predictions()

        input_vocab = tensorizers["tokens"].vocab

        class Model(jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.vocab = Vocabulary(input_vocab, unk_idx=input_vocab.idx[UNK])
                self.model = traced_model
                self.output_layer = output_layer
                self.pad_idx = jit.Attribute(input_vocab.idx[PAD], int)

            @jit.script_method
            def forward(self, tokens: List[List[str]]):
                word_ids = self.vocab.lookup_indices_2d(tokens)

                seq_lens = jit.annotate(List[int], [])

                for sentence in word_ids:
                    seq_lens.append(len(sentence))
                pad_to_length = list_max(seq_lens)
                for sentence in word_ids:
                    for _ in range(pad_to_length - len(sentence)):
                        sentence.append(self.pad_idx)

                logits = self.model((torch.tensor(word_ids), torch.tensor(seq_lens)))
                return self.output_layer(logits)

        return Model()

    def forward(
        self,
        tokens: Tuple[torch.Tensor, torch.Tensor],
        characters: Optional[torch.Tensor] = None,
        dicts: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        dense: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        word_tokens = None
        seq_lens = None
        if tokens is not None:
            word_tokens = tokens[0]
            seq_lens = tokens[1]
        inputs = [
            input_tensor
            for input_tensor in (word_tokens, characters, *(dicts or ()))
            if input_tensor is not None
        ]
        final_embedding = self.embedding(*inputs)
        representation = self.representation(final_embedding, seq_lens)
        # TODO: Unify all LSTM-style components to
        # not return state by default, then remove this
        if isinstance(representation, tuple):
            representation = representation[0]

        if dense:
            representation = torch.cat((representation, dense), 1)

        return self.decoder(representation)
