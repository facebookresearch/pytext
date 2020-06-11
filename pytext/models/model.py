#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from typing import Dict, List, Union

import torch
import torch.jit
import torch.nn as nn
from pytext.common.constants import Stage
from pytext.config.component import Component, ComponentType
from pytext.config.doc_classification import ModelInput
from pytext.config.field_config import FeatureConfig
from pytext.config.pytext_config import ConfigBase, ConfigBaseMeta
from pytext.config.serialize import _is_optional
from pytext.data import CommonMetadata
from pytext.data.tensorizers import Tensorizer
from pytext.models.module import create_module
from pytext.utils.file_io import PathManager
from pytext.utils.precision import maybe_float
from pytext.utils.usage import log_class_usage
from torch.jit import quantized

from .decoders import DecoderBase
from .embeddings import EmbeddingBase, EmbeddingList
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
            if getattr(t, "__origin__", "") is Union:
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
        log_class_usage(__class__)

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
        return self.output_layer.get_loss(logit, target, context)

    def get_pred(self, logit, target=None, context=None, *args):
        return self.output_layer.get_pred(logit, target, context)

    def save_modules(self, base_path: str = "", suffix: str = ""):
        """Save each sub-module in separate files for reusing later."""

        def save(module):
            save_path = getattr(module, "save_path", None)
            if save_path:
                path = os.path.join(base_path, module.save_path + suffix)
                print(f"Saving state of module {type(module).__name__} to {path} ...")
                with PathManager.open(path, "wb") as save_file:
                    if isinstance(module, torch.jit.ScriptModule):
                        module.save(save_file)
                    else:
                        torch.save(module.state_dict(), save_file)

        self.apply(save)

    def prepare_for_onnx_export_(self, **kwargs):
        """Make model exportable via ONNX trace."""

        def apply_prepare_for_onnx_export_(module):
            if module != self and hasattr(module, "prepare_for_onnx_export_"):
                module.prepare_for_onnx_export_(**kwargs)

        self.apply(apply_prepare_for_onnx_export_)

    def quantize(self):
        """Quantize the model during export."""
        # by default only quantize the linear modules, override this method if your
        # model wants other modules quantized.
        # By default we dynamic quantize Linear for PyText models.
        # Todo: we can also add quantized torch.nn.LSTM/GRU support in the future.
        torch.quantization.quantize_dynamic(
            self, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8, inplace=True
        )

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
    def train_batch(cls, model, batch, state=None):
        # This is a class method so that it works when model is a DistributedModel
        # wrapper. Otherwise the forward call here skips the DDP forward call.

        # Forward pass through the network.
        model_inputs = model.arrange_model_inputs(batch)
        model_context = model.arrange_model_context(batch)
        targets = model.arrange_targets(batch)
        model_outputs = model(*model_inputs)

        # Add stage to context.
        if state:
            if model_context is None:
                model_context = {"stage": state.stage}
            else:
                model_context["stage"] = state.stage

        # Compute loss and predictions.
        loss = maybe_float(model.get_loss(model_outputs, targets, model_context))
        predictions, scores = model.get_pred(model_outputs, context=model_context)

        # Pack results and return them.
        metric_data = (predictions, targets, scores, loss, model_inputs)
        return loss, metric_data

    def arrange_model_inputs(self, tensor_dict):
        # should raise NotImplementedError after migration is done
        pass

    def arrange_targets(self, tensor_dict):
        # should raise NotImplementedError after migration is done
        pass

    def arrange_model_context(self, tensor_dict):
        # should raise NotImplementedError after migration is done
        return None

    def onnx_trace_input(self, tensor_dict):
        # default behavior is the same as getting model inputs
        return self.arrange_model_inputs(tensor_dict)

    def caffe2_export(self, tensorizers, tensor_dict, path, export_onnx_path=None):
        pass

    def arrange_caffe2_model_inputs(self, tensor_dict):
        """
        Generate inputs for exported caffe2 model, default behavior is flatten the
        input tuples
        """
        model_inputs = self.arrange_model_inputs(tensor_dict)
        flat_model_inputs = []
        for model_input in model_inputs:
            if isinstance(model_input, tuple):
                flat_model_inputs.extend(model_input)
            else:
                flat_model_inputs.append(model_input)
        return flat_model_inputs

    def get_num_examples_from_batch(self, batch):
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
        # This config flag tells the model whether its parameters are being
        # loaded from saved state, hence it can skip certain initialization
        # steps such as loading pre-trained embeddings from file.
        #
        # TODO (geoffreygoh): Using config for such a purpose is really a hack,
        # and the alternative is either to pickle model objects directly so we
        # can skip initialization, or to pass this flag as an additional param
        # to create_model (which will involve changing from_config method of
        # every model in the repository). Clean this up once the above pickling
        # solution is fully explored.
        init_from_saved_state = False

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
        log_class_usage(__class__)

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
            decoder_inputs = inputs[-(self.decoder.num_decoder_modules) :]
        return self.decoder(
            *input_representation, *decoder_inputs
        )  # returned Tensor's dim = (batch_size, num_classes)
