#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, List, Optional, Union

import torch
from pytext.config.component import create_loss
from pytext.data.tensorizers import (
    ByteTokenTensorizer,
    FloatListTensorizer,
    LabelTensorizer,
    NumericLabelTensorizer,
    Tensorizer,
    TokenTensorizer,
    UidTensorizer,
)
from pytext.data.tokenizers import DoNothingTokenizer
from pytext.exporters.exporter import ModelExporter
from pytext.loss import BinaryCrossEntropyLoss, MultiLabelSoftMarginLoss
from pytext.models.decoders.mlp_decoder import DecoderBase, MLPDecoder
from pytext.models.embeddings import (
    CharacterEmbedding,
    EmbeddingBase,
    EmbeddingList,
    WordEmbedding,
)
from pytext.models.model import Model
from pytext.models.module import create_module
from pytext.models.output_layers import (
    ClassificationOutputLayer,
    OutputLayerBase,
    RegressionOutputLayer,
)
from pytext.models.output_layers.doc_classification_output_layer import (
    BinaryClassificationOutputLayer,
    MulticlassOutputLayer,
    MultiLabelOutputLayer,
)
from pytext.models.representations.bilstm_doc_attention import BiLSTMDocAttention
from pytext.models.representations.deepcnn import DeepCNNRepresentation
from pytext.models.representations.docnn import DocNNRepresentation
from pytext.models.representations.pure_doc_attention import PureDocAttention
from pytext.models.representations.representation_base import RepresentationBase
from pytext.torchscript.utils import (
    make_byte_inputs,
    make_sequence_lengths,
    pad_2d,
    truncate_tokens,
)
from pytext.torchscript.vocab import ScriptVocabulary
from pytext.utils.label import get_label_weights
from pytext.utils.usage import log_class_usage
from torch import jit


class DocModel(Model):
    """DocModel that's compatible with the new Model abstraction, which is responsible
    for describing which inputs it expects and arranging its input tensors."""

    __EXPANSIBLE__ = True

    class Config(Model.Config):
        class ModelInput(Model.Config.ModelInput):
            tokens: TokenTensorizer.Config = TokenTensorizer.Config()
            dense: Optional[FloatListTensorizer.Config] = None
            labels: LabelTensorizer.Config = LabelTensorizer.Config()

        inputs: ModelInput = ModelInput()
        embedding: WordEmbedding.Config = WordEmbedding.Config()
        representation: Union[
            PureDocAttention.Config,
            BiLSTMDocAttention.Config,
            DocNNRepresentation.Config,
            DeepCNNRepresentation.Config,
        ] = BiLSTMDocAttention.Config()
        decoder: MLPDecoder.Config = MLPDecoder.Config()
        output_layer: ClassificationOutputLayer.Config = (
            ClassificationOutputLayer.Config()
        )

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens, _ = tensor_dict["tokens"]
        model_inputs = (tokens, seq_lens)
        if "dense" in tensor_dict:
            model_inputs += (tensor_dict["dense"],)
        return model_inputs

    def arrange_targets(self, tensor_dict):
        return tensor_dict["labels"]

    def get_num_examples_from_batch(self, tensor_dict):
        targets = self.arrange_targets(tensor_dict)
        return len(targets)

    def get_export_input_names(self, tensorizers):
        res = ["tokens_vals", "tokens_lens"]
        if "dense" in tensorizers:
            res += ["float_vec_vals"]
        return res

    def get_export_output_names(self, tensorizers):
        return ["scores"]

    def vocab_to_export(self, tensorizers):
        return {"tokens_vals": list(tensorizers["tokens"].vocab)}

    def caffe2_export(self, tensorizers, tensor_dict, path, export_onnx_path=None):
        exporter = ModelExporter(
            ModelExporter.Config(),
            self.get_export_input_names(tensorizers),
            self.arrange_model_inputs(tensor_dict),
            self.vocab_to_export(tensorizers),
            self.get_export_output_names(tensorizers),
        )
        return exporter.export_to_caffe2(self, path, export_onnx_path=export_onnx_path)

    def torchscriptify(self, tensorizers, traced_model):
        output_layer = self.output_layer.torchscript_predictions()

        input_vocab = tensorizers["tokens"].vocab
        max_seq_len = tensorizers["tokens"].max_seq_len or -1
        scripted_tokenizer: Optional[jit.ScriptModule] = None
        try:
            scripted_tokenizer = tensorizers["tokens"].tokenizer.torchscriptify()
        except NotImplementedError:
            pass
        if scripted_tokenizer and isinstance(scripted_tokenizer, DoNothingTokenizer):
            scripted_tokenizer = None

        """
        The input tensor packing memory is allocated/cached for different shapes,
        and max sequence length will help to reduce the number of different tensor
        shapes. We noticed that the TorchScript model could use 25G for offline
        inference on CPU without using max_seq_len.
        """

        class Model(jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.vocab = ScriptVocabulary(
                    input_vocab,
                    input_vocab.get_unk_index(),
                    input_vocab.get_pad_index(),
                )
                self.model = traced_model
                self.output_layer = output_layer
                self.pad_idx = jit.Attribute(input_vocab.get_pad_index(), int)
                self.max_seq_len = jit.Attribute(max_seq_len, int)
                self.tokenizer = scripted_tokenizer

            @jit.script_method
            def forward(
                self,
                texts: Optional[List[str]] = None,
                multi_texts: Optional[List[List[str]]] = None,
                tokens: Optional[List[List[str]]] = None,
                languages: Optional[List[str]] = None,
            ):
                # PyTorch breaks with 2 'not None' checks right now.
                if texts is not None:
                    if tokens is not None:
                        raise RuntimeError("Can't set both tokens and texts")
                    if self.tokenizer is not None:
                        tokens = [
                            [t[0] for t in self.tokenizer.tokenize(text)]
                            for text in texts
                        ]

                if tokens is None:
                    raise RuntimeError("tokens is required")

                tokens = truncate_tokens(tokens, self.max_seq_len, self.vocab.pad_token)
                seq_lens = make_sequence_lengths(tokens)
                word_ids = self.vocab.lookup_indices_2d(tokens)
                word_ids = pad_2d(word_ids, seq_lens, self.pad_idx)
                logits = self.model(torch.tensor(word_ids), torch.tensor(seq_lens))
                return self.output_layer(logits)

        class ModelWithDenseFeat(jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.vocab = ScriptVocabulary(
                    input_vocab,
                    input_vocab.get_unk_index(),
                    input_vocab.get_pad_index(),
                )
                self.normalizer = tensorizers["dense"].normalizer
                self.model = traced_model
                self.output_layer = output_layer
                self.pad_idx = jit.Attribute(input_vocab.get_pad_index(), int)
                self.max_seq_len = jit.Attribute(max_seq_len, int)
                self.tokenizer = scripted_tokenizer

            @jit.script_method
            def forward(
                self,
                texts: Optional[List[str]] = None,
                multi_texts: Optional[List[List[str]]] = None,
                tokens: Optional[List[List[str]]] = None,
                languages: Optional[List[str]] = None,
                dense_feat: Optional[List[List[float]]] = None,
            ):
                # PyTorch breaks with 2 'not None' checks right now.
                if texts is not None:
                    if tokens is not None:
                        raise RuntimeError("Can't set both tokens and texts")
                    if self.tokenizer is not None:
                        tokens = [
                            [t[0] for t in self.tokenizer.tokenize(text)]
                            for text in texts
                        ]

                if tokens is None:
                    raise RuntimeError("tokens is required")
                if dense_feat is None:
                    raise RuntimeError("dense_feat is required")

                tokens = truncate_tokens(tokens, self.max_seq_len, self.vocab.pad_token)
                seq_lens = make_sequence_lengths(tokens)
                word_ids = self.vocab.lookup_indices_2d(tokens)
                word_ids = pad_2d(word_ids, seq_lens, self.pad_idx)
                dense_feat = self.normalizer.normalize(dense_feat)
                logits = self.model(
                    torch.tensor(word_ids),
                    torch.tensor(seq_lens),
                    torch.tensor(dense_feat, dtype=torch.float),
                )
                return self.output_layer(logits)

        return ModelWithDenseFeat() if "dense" in tensorizers else Model()

    @classmethod
    def create_embedding(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        return create_module(
            config.embedding,
            tensorizer=tensorizers["tokens"],
            init_from_saved_state=config.init_from_saved_state,
        )

    @classmethod
    def create_decoder(cls, config: Config, representation_dim: int, num_labels: int):
        num_decoder_modules = 0
        in_dim = representation_dim
        if hasattr(config.inputs, "dense") and config.inputs.dense:
            num_decoder_modules += 1
            in_dim += config.inputs.dense.dim
        decoder = create_module(config.decoder, in_dim=in_dim, out_dim=num_labels)
        decoder.num_decoder_modules = num_decoder_modules
        return decoder

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        labels = tensorizers["labels"].vocab
        if not labels:
            raise ValueError("Labels were not created, see preceding errors")

        embedding = cls.create_embedding(config, tensorizers)
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        decoder = cls.create_decoder(
            config, representation.representation_dim, len(labels)
        )

        label_weights = (
            get_label_weights(labels.idx, config.output_layer.label_weights)
            if config.output_layer.label_weights
            else None
        )
        loss = create_loss(config.output_layer.loss, weight=label_weights)

        if isinstance(loss, BinaryCrossEntropyLoss):
            output_layer_cls = BinaryClassificationOutputLayer
        elif isinstance(loss, MultiLabelSoftMarginLoss):
            output_layer_cls = MultiLabelOutputLayer
        else:
            output_layer_cls = MulticlassOutputLayer

        output_layer = output_layer_cls(list(labels), loss)
        return cls(embedding, representation, decoder, output_layer)


class ByteTokensDocumentModel(DocModel):
    """
    DocModel that receives both word IDs and byte IDs as inputs (concatenating
    word and byte-token embeddings to represent input tokens).
    """

    class Config(DocModel.Config):
        class ByteModelInput(DocModel.Config.ModelInput):
            token_bytes: ByteTokenTensorizer.Config = ByteTokenTensorizer.Config()

        inputs: ByteModelInput = ByteModelInput()
        byte_embedding: CharacterEmbedding.Config = CharacterEmbedding.Config()

    @classmethod
    def create_embedding(cls, config, tensorizers: Dict[str, Tensorizer]):
        word_tensorizer = config.inputs.tokens
        byte_tensorizer = config.inputs.token_bytes
        assert word_tensorizer.column == byte_tensorizer.column

        word_embedding = create_module(
            config.embedding,
            tensorizer=tensorizers["tokens"],
            init_from_saved_state=config.init_from_saved_state,
        )
        byte_embedding = create_module(
            config.byte_embedding, vocab_size=ByteTokenTensorizer.NUM_BYTES
        )
        return EmbeddingList([word_embedding, byte_embedding], concat=True)

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens, _ = tensor_dict["tokens"]
        token_bytes, byte_seq_lens, _ = tensor_dict["token_bytes"]
        assert (seq_lens == byte_seq_lens).all().item()
        model_inputs = tokens, token_bytes, seq_lens
        if "dense" in tensor_dict:
            model_inputs += (tensor_dict["dense"],)
        return model_inputs

    def get_export_input_names(self, tensorizers):
        names = ["tokens_vals", "token_bytes", "tokens_lens"]
        if "dense" in tensorizers:
            names.append("float_vec_vals")
        return names

    def torchscriptify(self, tensorizers, traced_model):
        output_layer = self.output_layer.torchscript_predictions()
        max_seq_len = tensorizers["tokens"].max_seq_len or -1
        max_byte_len = tensorizers["token_bytes"].max_byte_len
        byte_offset_for_non_padding = tensorizers["token_bytes"].offset_for_non_padding
        input_vocab = tensorizers["tokens"].vocab

        class Model(jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.vocab = ScriptVocabulary(
                    input_vocab,
                    input_vocab.get_unk_index(),
                    input_vocab.get_pad_index(),
                )
                self.max_seq_len = jit.Attribute(max_seq_len, int)
                self.max_byte_len = jit.Attribute(max_byte_len, int)
                self.byte_offset_for_non_padding = jit.Attribute(
                    byte_offset_for_non_padding, int
                )
                self.pad_idx = jit.Attribute(input_vocab.get_pad_index(), int)
                self.model = traced_model
                self.output_layer = output_layer

            @jit.script_method
            def forward(
                self,
                texts: Optional[List[str]] = None,
                multi_texts: Optional[List[List[str]]] = None,
                tokens: Optional[List[List[str]]] = None,
                languages: Optional[List[str]] = None,
            ):
                if tokens is None:
                    raise RuntimeError("tokens is required")

                tokens = truncate_tokens(tokens, self.max_seq_len, self.vocab.pad_token)
                seq_lens = make_sequence_lengths(tokens)
                word_ids = self.vocab.lookup_indices_2d(tokens)
                word_ids = pad_2d(word_ids, seq_lens, self.pad_idx)
                token_bytes, _ = make_byte_inputs(
                    tokens, self.max_byte_len, self.byte_offset_for_non_padding
                )
                logits = self.model(
                    torch.tensor(word_ids), token_bytes, torch.tensor(seq_lens)
                )
                return self.output_layer(logits)

        class ModelWithDenseFeat(jit.ScriptModule):
            def __init__(self):
                super().__init__()
                self.vocab = ScriptVocabulary(
                    input_vocab,
                    input_vocab.get_unk_index(),
                    input_vocab.get_pad_index(),
                )
                self.normalizer = tensorizers["dense"].normalizer
                self.max_seq_len = jit.Attribute(max_seq_len, int)
                self.max_byte_len = jit.Attribute(max_byte_len, int)
                self.byte_offset_for_non_padding = jit.Attribute(
                    byte_offset_for_non_padding, int
                )
                self.pad_idx = jit.Attribute(input_vocab.get_pad_index(), int)
                self.model = traced_model
                self.output_layer = output_layer

            @jit.script_method
            def forward(
                self,
                texts: Optional[List[str]] = None,
                multi_texts: Optional[List[List[str]]] = None,
                tokens: Optional[List[List[str]]] = None,
                languages: Optional[List[str]] = None,
                dense_feat: Optional[List[List[float]]] = None,
            ):
                if tokens is None:
                    raise RuntimeError("tokens is required")
                if dense_feat is None:
                    raise RuntimeError("dense_feat is required")

                tokens = truncate_tokens(tokens, self.max_seq_len, self.vocab.pad_token)
                seq_lens = make_sequence_lengths(tokens)
                word_ids = self.vocab.lookup_indices_2d(tokens)
                word_ids = pad_2d(word_ids, seq_lens, self.pad_idx)
                token_bytes, _ = make_byte_inputs(
                    tokens, self.max_byte_len, self.byte_offset_for_non_padding
                )
                dense_feat = self.normalizer.normalize(dense_feat)
                logits = self.model(
                    torch.tensor(word_ids),
                    token_bytes,
                    torch.tensor(seq_lens),
                    torch.tensor(dense_feat, dtype=torch.float),
                )
                return self.output_layer(logits)

        return ModelWithDenseFeat() if "dense" in tensorizers else Model()


class DocRegressionModel(DocModel):
    """
    Model that's compatible with the new Model abstraction, and is configured for
    regression tasks (specifically for labels, predictions, and loss).
    """

    class Config(DocModel.Config):
        class RegressionModelInput(DocModel.Config.ModelInput):
            tokens: TokenTensorizer.Config = TokenTensorizer.Config()
            labels: NumericLabelTensorizer.Config = NumericLabelTensorizer.Config()

        inputs: RegressionModelInput = RegressionModelInput()
        output_layer: RegressionOutputLayer.Config = RegressionOutputLayer.Config()

    @classmethod
    def from_config(cls, config: Config, tensorizers: Dict[str, Tensorizer]):
        embedding = cls.create_embedding(config, tensorizers)
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        decoder = create_module(
            config.decoder, in_dim=representation.representation_dim, out_dim=1
        )
        output_layer = RegressionOutputLayer.from_config(config.output_layer)
        return cls(embedding, representation, decoder, output_layer)


class PersonalizedDocModel(DocModel):
    """
    DocModel that includes a user embedding which learns user features to produce
    personalized prediction. In this class, user-embedding is fed directly to
    the decoder (i.e., does not go through the encoders).
    """

    class Config(DocModel.Config):
        class PersonalizedModelInput(DocModel.Config.ModelInput):
            uid: Optional[UidTensorizer.Config] = UidTensorizer.Config()

        inputs: PersonalizedModelInput = PersonalizedModelInput()
        # user_embedding is a representation for a user and is jointly trained
        # with the model. Consider user ids as "words" to reuse WordEmbedding class.
        user_embedding: WordEmbedding.Config = WordEmbedding.Config()

    @classmethod
    def from_config(cls, config, tensorizers: Dict[str, Tensorizer]):
        model = super().from_config(config, tensorizers)

        user_embedding = create_module(
            config.user_embedding,
            tensorizer=tensorizers["uid"],
            init_from_saved_state=config.init_from_saved_state,
        )
        # Init user embeddings to be a same vector because we assume user features
        # are not too different from each other.
        emb_shape = user_embedding.word_embedding.weight.data.shape
        with torch.no_grad():
            user_embedding.word_embedding.weight.copy_(
                torch.rand(emb_shape[1]).repeat(emb_shape[0], 1)
            )

        labels = tensorizers["labels"].vocab
        decoder = cls.create_decoder(
            config,
            model.representation.representation_dim + user_embedding.embedding_dim,
            len(labels),
        )

        return cls(
            model.embedding,
            model.representation,
            decoder,
            model.output_layer,
            user_embedding,
        )

    def __init__(
        self,
        embedding: EmbeddingBase,
        representation: RepresentationBase,
        decoder: DecoderBase,
        output_layer: OutputLayerBase,
        user_embedding: Optional[EmbeddingBase] = None,
    ):
        super().__init__(embedding, representation, decoder, output_layer)
        self.user_embedding = user_embedding
        log_class_usage(__class__)

    def arrange_model_inputs(self, tensor_dict):
        model_inputs = super().arrange_model_inputs(tensor_dict)
        model_inputs += (tensor_dict["uid"],)
        return model_inputs

    def vocab_to_export(self, tensorizers):
        export_vocab = super().vocab_to_export(tensorizers)
        export_vocab["uid"] = list(tensorizers["uid"].vocab)
        return export_vocab

    def get_export_input_names(self, tensorizers):
        export_inputs_names = super().get_export_input_names(tensorizers)
        export_inputs_names += ["uid"]
        return export_inputs_names

    def torchscriptify(self, tensorizers, traced_model):
        raise NotImplementedError("This model is not jittable yet.")

    def forward(self, *inputs) -> List[torch.Tensor]:
        # Override forward() to include user_embedding layer explictily to the
        # computation of the forward propagation.
        embedding_input = inputs[: self.embedding.num_emb_modules]
        token_emb = self.embedding(*embedding_input)

        inputs, user_ids = inputs[:-1], inputs[-1]
        other_input = inputs[
            self.embedding.num_emb_modules : len(inputs)
            - self.decoder.num_decoder_modules
        ]
        input_representation = self.representation(token_emb, *other_input)

        # Some LSTM-based representations return states as (h0, c0).
        if isinstance(input_representation[-1], tuple):
            input_representation = input_representation[0]

        user_embeddings = self.user_embedding(user_ids)
        # Reduce dim from (batch_size, 1, emb_dim) to (batch_size, emb_dim).
        if user_embeddings.dim() == 3:
            user_embeddings = user_embeddings.squeeze(dim=1)

        decoder_inputs: tuple = ()
        if self.decoder.num_decoder_modules:
            decoder_inputs = inputs[-(self.decoder.num_decoder_modules) :]

        return self.decoder(
            input_representation, user_embeddings, *decoder_inputs
        )  # Return Tensor's dim = (batch_size, num_classes).
