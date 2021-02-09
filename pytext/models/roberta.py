#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional, Tuple

import torch
from pytext import resources
from pytext.common.constants import Stage
from pytext.config import ConfigBase
from pytext.data.roberta_tensorizer import (
    RoBERTaTensorizer,
    RoBERTaTokenLevelTensorizer,
)
from pytext.data.tensorizers import (
    FloatListTensorizer,
    LabelTensorizer,
    NumericLabelTensorizer,
    Tensorizer,
)
from pytext.models.bert_classification_models import NewBertModel
from pytext.models.bert_regression_model import NewBertRegressionModel
from pytext.models.decoders.mlp_decoder import MLPDecoder
from pytext.models.model import BaseModel
from pytext.models.module import Module, create_module
from pytext.models.output_layers import WordTaggingOutputLayer
from pytext.models.representations.transformer import (
    MultiheadLinearAttention,
    QuantizedMultiheadLinearAttention,
    MultiheadSelfAttention,
    PostEncoder,
    SELFIETransformer,
    SentenceEncoder,
    Transformer,
    TransformerLayer,
)
from pytext.models.representations.transformer.transformer import (
    DEFAULT_MAX_SEQUENCE_LENGTH,
)
from pytext.models.representations.transformer_sentence_encoder_base import (
    PoolingMethod,
    TransformerSentenceEncoderBase,
)
from pytext.models.utils import normalize_embeddings
from pytext.torchscript.module import (
    ScriptPyTextEmbeddingModuleIndex,
    ScriptPyTextModule,
    ScriptPyTextModuleWithDense,
)
from pytext.utils.file_io import PathManager
from pytext.utils.usage import log_class_usage
from torch import nn
from torch.quantization import convert_jit, get_default_qconfig, prepare_jit
from torch.serialization import default_restore_location

from .r3f_models import R3FConfigOptions, R3FPyTextMixin


def init_params(module):
    """Initialize the RoBERTa weights for pre-training from scratch."""

    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class RoBERTaEncoderBase(TransformerSentenceEncoderBase):
    __EXPANSIBLE__ = True

    class Config(TransformerSentenceEncoderBase.Config):
        pass

    def _encoder(self, inputs, *args):
        # NewBertModel expects the output as a tuple and grabs the first element
        tokens, _, _, _ = inputs
        full_representation = (
            self.encoder(tokens, args) if len(args) > 0 else self.encoder(tokens)
        )
        sentence_rep = full_representation[-1][:, 0, :]
        return full_representation, sentence_rep


class RoBERTaEncoderJit(RoBERTaEncoderBase):
    """A TorchScript RoBERTa implementation"""

    class Config(RoBERTaEncoderBase.Config):
        pretrained_encoder: Module.Config = Module.Config(
            load_path=resources.roberta.PUBLIC
        )

    def __init__(self, config: Config, output_encoded_layers: bool, **kwarg) -> None:
        config.pretrained_encoder.load_path = (
            resources.roberta.RESOURCE_MAP[config.pretrained_encoder.load_path]
            if config.pretrained_encoder.load_path in resources.roberta.RESOURCE_MAP
            else config.pretrained_encoder.load_path
        )
        super().__init__(config, output_encoded_layers=output_encoded_layers)
        assert config.pretrained_encoder.load_path, "Load path cannot be empty."
        self.encoder = create_module(config.pretrained_encoder)
        self.representation_dim = self.encoder.encoder.token_embedding.weight.size(-1)
        log_class_usage(__class__)

    def _embedding(self):
        # used to tie weights in MaskedLM model
        return self.encoder.encoder.token_embedding


class RoBERTaEncoder(RoBERTaEncoderBase):
    """A PyTorch RoBERTa implementation"""

    class Config(RoBERTaEncoderBase.Config):
        embedding_dim: int = 768
        vocab_size: int = 50265
        num_encoder_layers: int = 12
        num_attention_heads: int = 12
        model_path: str = (
            "manifold://pytext_training/tree/static/models/roberta_base_torch.pt"
        )
        # Loading the state dict of the model depends on whether the model was
        # previously finetuned in PyText or not. If it was finetuned then we
        # dont need to translate the state dict and can just load it`
        # directly.
        is_finetuned: bool = False
        max_seq_len: int = DEFAULT_MAX_SEQUENCE_LENGTH
        # Fine-tune bias parameters only (https://nlp.biu.ac.il/~yogo/bitfit.pdf)
        use_bias_finetuning: bool = False
        # Linformer hyperparameters
        use_linformer_encoder: bool = False
        linformer_compressed_ratio: int = 4
        linformer_quantize: bool = False
        export_encoder: bool = False
        variable_size_embedding: bool = True
        use_selfie_encoder: bool = False

    def __init__(self, config: Config, output_encoded_layers: bool, **kwarg) -> None:
        super().__init__(config, output_encoded_layers=output_encoded_layers)

        # map to the real model_path
        config.model_path = (
            resources.roberta.RESOURCE_MAP[config.model_path]
            if config.model_path in resources.roberta.RESOURCE_MAP
            else config.model_path
        )
        # assert config.pretrained_encoder.load_path, "Load path cannot be empty."
        # sharing compression across each layers

        # create compress layer if use linear multihead attention
        if config.use_linformer_encoder:
            compress_layer = nn.Linear(
                config.max_seq_len - 2,
                (config.max_seq_len - 2) // config.linformer_compressed_ratio,
            )

        self.use_selfie_encoder = config.use_selfie_encoder

        if config.use_linformer_encoder:
            if config.linformer_quantize:
                layers = [
                    TransformerLayer(
                        embedding_dim=config.embedding_dim,
                        attention=QuantizedMultiheadLinearAttention(
                            embed_dim=config.embedding_dim,
                            num_heads=config.num_attention_heads,
                            compress_layer=compress_layer,
                        ),
                    )
                    for _ in range(config.num_encoder_layers)
                ]
            else:
                layers = [
                    TransformerLayer(
                        embedding_dim=config.embedding_dim,
                        attention=MultiheadLinearAttention(
                            embed_dim=config.embedding_dim,
                            num_heads=config.num_attention_heads,
                            compress_layer=compress_layer,
                        ),
                    )
                    for _ in range(config.num_encoder_layers)
                ]
        else:
            layers = [
                TransformerLayer(
                    embedding_dim=config.embedding_dim,
                    attention=MultiheadSelfAttention(
                        embed_dim=config.embedding_dim,
                        num_heads=config.num_attention_heads,
                    ),
                )
                for _ in range(config.num_encoder_layers)
            ]

        self.encoder = (
            SentenceEncoder(
                transformer=Transformer(
                    vocab_size=config.vocab_size,
                    embedding_dim=config.embedding_dim,
                    layers=layers,
                    max_seq_len=config.max_seq_len,
                )
            )
            if not self.use_selfie_encoder
            else PostEncoder(
                transformer=SELFIETransformer(
                    vocab_size=config.vocab_size,
                    embedding_dim=config.embedding_dim,
                    layers=layers,
                    max_seq_len=config.max_seq_len,
                )
            )
        )
        self.apply(init_params)
        if config.model_path:
            with PathManager.open(config.model_path, "rb") as f:
                roberta_state = torch.load(
                    f, map_location=lambda s, l: default_restore_location(s, "cpu")
                )
            # In case the model has previously been loaded in PyText and finetuned,
            # then we dont need to do the special state dict translation. Load
            # it directly
            if not config.is_finetuned:
                self.encoder.load_roberta_state_dict(roberta_state["model"])
            else:
                self.load_state_dict(roberta_state)

        if config.use_bias_finetuning:
            for (n, p) in self.encoder.named_parameters():
                # "encoder.transformer.layers.0.attention.input_projection.weight" -> false
                # "encoder.transformer.layers.0.attention.input_projection.bias" -> true
                if n.split(".")[-1] != "bias":
                    p.requires_grad_(False)

        self.export_encoder = config.export_encoder
        self.variable_size_embedding = config.variable_size_embedding
        self.use_linformer_encoder = config.use_linformer_encoder
        log_class_usage(__class__)

    def _embedding(self):
        # used to tie weights in MaskedLM model
        return self.encoder.transformer.token_embedding

    def forward(
        self, input_tuple: Tuple[torch.Tensor, ...], *args
    ) -> Tuple[torch.Tensor, ...]:

        encoded_layers, pooled_output = (
            self._encoder(input_tuple, args[0])
            if self.use_selfie_encoder
            else self._encoder(input_tuple)
        )

        pad_mask = input_tuple[1]

        if self.pooling != PoolingMethod.CLS_TOKEN:
            pooled_output = self._pool_encoded_layers(encoded_layers, pad_mask)

        if self.projection:
            pooled_output = self.projection(pooled_output).tanh()

        if pooled_output is not None:
            pooled_output = self.output_dropout(pooled_output)
            if self.normalize_output_rep:
                pooled_output = normalize_embeddings(pooled_output)

        output = []
        if self.output_encoded_layers:
            output.append(encoded_layers)
        if self.pooling != PoolingMethod.NO_POOL:
            output.append(pooled_output)
        return tuple(output)


class RoBERTa(NewBertModel):
    class Config(NewBertModel.Config):
        class InputConfig(ConfigBase):
            tokens: RoBERTaTensorizer.Config = RoBERTaTensorizer.Config()
            dense: Optional[FloatListTensorizer.Config] = None
            labels: LabelTensorizer.Config = LabelTensorizer.Config()

        inputs: InputConfig = InputConfig()
        encoder: RoBERTaEncoderBase.Config = RoBERTaEncoderJit.Config()

    def trace(self, inputs):
        if self.encoder.export_encoder:
            return torch.jit.trace(self.encoder, inputs)
        else:
            return torch.jit.trace(self, inputs)

    def torchscriptify(self, tensorizers, traced_model):
        """Using the traced model, create a ScriptModule which has a nicer API that
        includes generating tensors from simple data types, and returns classified
        values according to the output layer (eg. as a dict mapping class name to score)
        """
        script_tensorizer = tensorizers["tokens"].torchscriptify()
        if self.encoder.export_encoder:
            return ScriptPyTextEmbeddingModuleIndex(
                traced_model, script_tensorizer, index=0
            )
        else:
            if "dense" in tensorizers:
                return ScriptPyTextModuleWithDense(
                    model=traced_model,
                    output_layer=self.output_layer.torchscript_predictions(),
                    tensorizer=script_tensorizer,
                    normalizer=tensorizers["dense"].normalizer,
                )
            else:
                return ScriptPyTextModule(
                    model=traced_model,
                    output_layer=self.output_layer.torchscript_predictions(),
                    tensorizer=script_tensorizer,
                )

    def graph_mode_quantize(
        self,
        inputs,
        data_loader,
        calibration_num_batches=64,
        qconfig_dict=None,
        force_quantize=False,
    ):
        """Quantize the model during export with graph mode quantization."""
        if force_quantize:
            trace = self.trace(inputs)
            if not qconfig_dict:
                qconfig_dict = {"": get_default_qconfig("fbgemm")}
            prepare_m = prepare_jit(trace, qconfig_dict, inplace=False)
            prepare_m.eval()
            with torch.no_grad():
                for i, (_, batch) in enumerate(data_loader):
                    print("Running calibration with batch {}".format(i))
                    input_data = self.onnx_trace_input(batch)
                    prepare_m(*input_data)
                    if i == calibration_num_batches - 1:
                        break
            trace = convert_jit(prepare_m, inplace=True)
        else:
            super().quantize()
            trace = self.trace(inputs)

        return trace


class SELFIE(RoBERTa):
    class Config(RoBERTa.Config):
        use_selfie: bool = True

    def forward(
        self, encoder_inputs: Tuple[torch.Tensor, ...], *args
    ) -> List[torch.Tensor]:
        if self.encoder.output_encoded_layers:
            # if encoded layers are returned, discard them
            representation = self.encoder(encoder_inputs, args[0])[1]
        else:
            representation = self.encoder(encoder_inputs, args[0])[0]
        return self.decoder(representation)


class RoBERTaRegression(NewBertRegressionModel):
    class Config(NewBertRegressionModel.Config):
        class RegressionModelInput(ConfigBase):
            tokens: RoBERTaTensorizer.Config = RoBERTaTensorizer.Config()
            labels: NumericLabelTensorizer.Config = NumericLabelTensorizer.Config()

        inputs: RegressionModelInput = RegressionModelInput()
        encoder: RoBERTaEncoderBase.Config = RoBERTaEncoderJit.Config()

    def torchscriptify(self, tensorizers, traced_model):
        """Using the traced model, create a ScriptModule which has a nicer API that
        includes generating tensors from simple data types, and returns classified
        values according to the output layer (eg. as a dict mapping class name to score)
        """
        script_tensorizer = tensorizers["tokens"].torchscriptify()
        return ScriptPyTextModule(
            model=traced_model,
            output_layer=self.output_layer.torchscript_predictions(),
            tensorizer=script_tensorizer,
        )


class RoBERTaWordTaggingModel(BaseModel):
    """
    Single Sentence Token-level Classification Model using XLM.
    """

    class Config(BaseModel.Config):
        class WordTaggingInputConfig(ConfigBase):
            tokens: RoBERTaTokenLevelTensorizer.Config = (
                RoBERTaTokenLevelTensorizer.Config()
            )

        inputs: WordTaggingInputConfig = WordTaggingInputConfig()
        encoder: RoBERTaEncoderBase.Config = RoBERTaEncoderJit.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: WordTaggingOutputLayer.Config = WordTaggingOutputLayer.Config()

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        label_vocab = tensorizers["tokens"].labels_vocab
        vocab = tensorizers["tokens"].vocab

        encoder = create_module(
            config.encoder,
            output_encoded_layers=True,
            padding_idx=vocab.get_pad_index(),
            vocab_size=vocab.__len__(),
        )
        decoder = create_module(
            config.decoder, in_dim=encoder.representation_dim, out_dim=len(label_vocab)
        )
        output_layer = create_module(config.output_layer, labels=label_vocab)
        return cls(encoder, decoder, output_layer)

    def __init__(self, encoder, decoder, output_layer, stage=Stage.TRAIN) -> None:
        super().__init__(stage=stage)
        self.encoder = encoder
        self.decoder = decoder
        self.module_list = [encoder, decoder]
        self.output_layer = output_layer
        self.stage = stage
        log_class_usage(__class__)

    def arrange_model_inputs(self, tensor_dict):
        tokens, pad_mask, segment_labels, positions, _ = tensor_dict["tokens"]
        model_inputs = (tokens, pad_mask, segment_labels, positions)
        return (model_inputs,)

    def arrange_targets(self, tensor_dict):
        _, _, _, _, labels = tensor_dict["tokens"]
        return labels

    def forward(self, encoder_inputs: Tuple[torch.Tensor, ...], *args) -> torch.Tensor:
        # The encoder outputs a list of representations for each token where
        # every element of the list corresponds to a layer in the transformer.
        # We extract and pass the representations associated with the last layer
        # of the transformer.
        representation = self.encoder(encoder_inputs)[0][-1]
        return self.decoder(representation, *args)


class RoBERTaR3F(RoBERTa, R3FPyTextMixin):
    class Config(RoBERTa.Config):
        r3f_options: R3FConfigOptions = R3FConfigOptions()

    def get_embedding_module(self, *args, **kwargs):
        return self.encoder.encoder.transformer.token_embedding

    def original_forward(self, *args, **kwargs):
        return RoBERTa.forward(self, *args, **kwargs)

    def get_sample_size(self, model_inputs, targets):
        return targets.size(0)

    def __init__(
        self, encoder, decoder, output_layer, r3f_options, stage=Stage.TRAIN
    ) -> None:
        RoBERTa.__init__(self, encoder, decoder, output_layer, stage=stage)
        R3FPyTextMixin.__init__(self, r3f_options)

    def forward(self, *args, use_r3f: bool = False, **kwargs):
        return R3FPyTextMixin.forward(self, *args, use_r3f=use_r3f, **kwargs)

    @classmethod
    def train_batch(cls, model, batch, state=None):
        return R3FPyTextMixin.train_batch(model=model, batch=batch, state=state)
