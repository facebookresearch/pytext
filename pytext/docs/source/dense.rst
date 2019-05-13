Using External Dense Features
=============================

Sometime you want to add external features to augment the inputs to your model. For example, if you want to classify a text that has an image associated to it, you might want to process the image separately and use features of this image along with the text to help the classifier. Those features are added in the input data as one extra field (column) and should look like a list of floats (json).

Let's look at a simple example, first without the dense feature, then we add dense features.

Example: Simple Model
---------------------

First, here's an example of a simple classifier that uses just the text and no dense features. (This is only showing the relevant parts of the model code for simplicity.)

.. code:: python

  class MyModel(Model):
    class Config(Model.Config):
      class ModelInput(Model.Config.InputConfig):
        tokens: TokenTensorizer.Config = TokenTensorizer.Config()
        labels: LabelTensorizer.Config = LabelTensorizer.Config()

      inputs: ModelInput = ModelInput()
      token_embedding: WordEmbedding.Config = WordEmbedding.Config()

      representation: RepresentationBase.Config = DocNNRepresentation.Config()
      decoder: DecoderBase.Config = MLPDecoder.Config()
      output_layer: OutputLayerBase.Config = ClassificationOutputLayer.Config()

    def from_config(cls, config, tensorizers):
      token_embedding = create_module(config.token_embedding, tensorizer=tensorizers["tokens"])
      representation = create_module(config.representation, embed_dim=token_embedding.embedding_dim)
      labels = tensorizers["labels"].vocab
      decoder = create_module(
          config.decoder,
          in_dim=representation.representation_dim
          out_dim=len(labels),
      )
      output_layer = create_module(config.output_layer, labels=labels)
      return cls(token_embedding, representation, decoder, output_layer)

    def arrange_model_inputs(self, tensor_dict):
      return (tensor_dict["tokens"],)

    def forward(
        self,
        tokens_in: Tuple[torch.Tensor, torch.Tensor],
    ) -> List[torch.Tensor]:
          word_tokens, seq_lens = tokens
          embedding_out = self.embedding(word_tokens)
          representation_out = self.representation(embedding_out, seq_lens)
          return self.decoder(representation_out)

Example: Simple Model With Dense Features
-----------------------------------------

To use the dense features, you will typically write your model to use them directly in the decoder, bypassing the embeddings and representation stages that process the text part of your inputs. Here's the same example again, this time with the dense features added (see lines marked with <--).

.. code:: python

  class MyModel(Model):
    class Config(Model.Config):
      class ModelInput(Model.Config.InputConfig):
        tokens: TokenTensorizer.Config = TokenTensorizer.Config()
        dense: FloatListTensorizer.Config = FloatListTensorizer.Config()  # <--
        labels: LabelTensorizer.Config = LabelTensorizer.Config()

      inputs: ModelInput = ModelInput()
      token_embedding: WordEmbedding.Config = WordEmbedding.Config()

      representation: RepresentationBase.Config = DocNNRepresentation.Config()
      decoder: DecoderBase.Config = MLPDecoder.Config()
      output_layer: OutputLayerBase.Config = ClassificationOutputLayer.Config()

    def from_config(cls, config, tensorizers):
      token_embedding = create_module(config.token_embedding, tensorizer=tensorizers["tokens"])
      representation = create_module(config.representation, embed_dim=token_embedding.embedding_dim)
      dense_dim = tensorizers["dense"].out_dim    # <--
      labels = tensorizers["labels"].vocab
      decoder = create_module(
          config.decoder,
          in_dim=representation.representation_dim + dense_dim    # <--
          out_dim=len(labels),
      )
      output_layer = create_module(config.output_layer, labels=labels)
      return cls(token_embedding, representation, decoder, output_layer)

    def arrange_model_inputs(self, tensor_dict):
      return (tensor_dict["tokens"], tensor_dict["dense"])  # <--

    def forward(
        self,
        tokens_in: Tuple[torch.Tensor, torch.Tensor],
        dense_in: torch.Tensor,    # <--
    ) -> List[torch.Tensor]:
          word_tokens, seq_lens = tokens
          embedding_out = self.embedding(word_tokens)
          representation_out = self.representation(embedding_out, seq_lens)
          representation_out = torch.cat((representation_out, dense_in), 1)    # <--
          return self.decoder(representation_out)
