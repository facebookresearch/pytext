Creating A New Model
====================

PyText uses a :class:`~Model` class as a central place to define components for data processing, model training, etc. and wire up those components.

In this tutorial, we will create a word tagging model for the ATIS dataset. The format of the ATIS dataset is explained in the :doc:`datasource_tutorial`, so we will not repeat it here. We are going to create a similar data source that uses the slot tagging information rather than the intent information. We won't describe in detail how this data source is created but you can look at the :doc:`datasource_tutorial`, and the full source code for this tutorial in ``demo/my_tagging`` for more information.

This model will predict a "slot", also called "tag" or "label", for each word in the utterance, using the `IOB2 format <https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging>`_), where the O tag is used for Outside (no match), B- for Beginning and I- for Inside (continuation). Here's an example:

.. code-block:: console

  {
    "text": "please list the flights from newark to los angeles",
    "slots": "O O O O O B-fromloc.city_name O B-toloc.city_name I-toloc.city_name"
  }


1. The Components
-----------------

The first step is to specify the components used in this model by listing them in the Config class, the corresponding ``from_config`` function, and the constructor ``__init__``.

Thanks to the modular nature of PyText, we can simply use many included common components, such as :class:`TokenTensorizer`, :class:`WordEmbedding`, :class:`BiLSTMSlotAttention` and :class:`MLPDecoder`. Since we're also using the common pattern of `embedding` -> `representation` -> `decoder` -> `output_layer`, we use :class:`Model` as a base class, so we don't need to write ``__init__``.

ModelInput defines how the data that is read will be transformed into tensors. This is done using a :class:`Tensorizer`. These components take one or several columns (often strings) from each input row and create the corresponding numeric features in a properly padded tensor. The tensorizers will to be initialized first, and in this step they will often parse the training data to create their :class:`Vocabulary`.

In our case, the utterance is in the column "text" (which is the default column name for this tensorizer), and is composed of tokens (words), so we can use the :class:`TokenTensorizer`. The :class:`Vocabulary` will be created from all the utterances.

The slots are also composed of tokens: the IOB2 tags. We can also use :class:`TokenTensorizer` for the column "slots". This :class:`Vocabulary` will be the list of IOB2 tags found in the "slots" column of the training data. This is a different column name, so we specify it.

.. code:: python

  class MyTaggingModel(Model):
      class Config(ConfigBase):
          class ModelInput(Model.Config.ModelInput):
              tokens: TokenTensorizer.Config = TokenTensorizer.Config()
              slots: TokenTensorizer.Config = TokenTensorizer.Config(column="slots")

          inputs: ModelInput = ModelInput()
          embedding: WordEmbedding.Config = WordEmbedding.Config()
          representation: BiLSTMSlotAttention.Config = BiLSTMSlotAttention.Config,
          decoder: MLPDecoder.Config = MLPDecoder.Config()
          output_layer: MyTaggingOutputLayer.Config = MyTaggingOutputLayer.Config()


2. from_config method
---------------------

``from_config`` is where the components are created with the proper parameters. Some come from the Config (passed by the user in json format), some use the default values, others are dicated by the model's architecture so that the different components fit with each other. For example, the representation layer needs to know the dimension of the embeddings it will receive, the decoder needs to know the dimension of the representation layer before it and the size of the slots vocab to output.

In this model, we only need one embedding: the one of the tokens. The slots don't have embeddings because while they are listed as input (in ModelInput), they are actually outputs and the will be used in the output layer. (During training, they are inputs as true values.)

.. code:: python

    @classmethod
    def from_config(cls, config, tensorizers):
        embedding = create_module(config.embedding, tensorizer=tensorizers["tokens"])
        representation = create_module(
            config.representation, embed_dim=embedding.embedding_dim
        )
        slots = tensorizers["slots"].vocab
        decoder = create_module(
            config.decoder,
            in_dim=representation.representation_dim,
            out_dim=len(slots),
        )
        output_layer = MyTaggingOutputLayer(slots, CrossEntropyLoss(None))
        # call __init__ constructor from super class Model
        return cls(embedding, representation, decoder, output_layer)


3. Forward method
-----------------

The ``forward`` method contains the execution logic calling each of those components and passing the results of one to the next. It will be called for every row transformed into tensors.

:class:`TokenTensorizer` returns the tensor for the tokens themselves and also the sequence length, which is the number of tokens in the utterances. This is because we need to pad the tensors in a batch to give them all the same dimensions, and LSTM-based reprentations need to differentiate the padding from the actual tokens.

.. code:: python

    def forward(
        self,
        word_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> List[torch.Tensor]:
        # fetch embeddings for the tokens in the utterance
        embedding = self.embedding(word_tokens)

        # pass the embeddings to the BiLSTMSlotAttention layer.
        # LSTM-based representations also need seq_lens.
        representation = self.representation(embedding, seq_lens)

        # some LSTM representations return extra tensors, we don't use those.
        if isinstance(representation, tuple):
            representation = representation[0]

        # finally run the results through the decoder
        return self.decoder(representation)


4. Complete MyTaggingModel
--------------------------

To finish this class, we need to define a few more functions.

All the inputs are placed in a python dict where the key is the name of the tensorizer as defined in ModelInput, and the value is the tensor for this input row.

First, we define how the inputs will be passed to the ``forward`` function in ``arrange_model_inputs``. In our case, the only input passed to the ``forward`` function is the tensors from the "tokens" input. As explained above, :class:`TokenTensorizer` returns 2 tensors: the tokens and the sequence length. (Actually it returns 3 tensors, we'll ignore the 3rd one, the token ranges, in this tutorial)

Then we define ``arrange_targets``, which is doing something similar for the targets, which are passed to the loss function during training. In our case, it's the "slots" tensorizer doing that. The padding value can be passed to the loss function (unlike LSTM representations), so we only need the first tensor.

.. code:: python

    def arrange_model_inputs(self, tensor_dict):
        tokens, seq_lens, _ = tensor_dict["tokens"]
        return (tokens, seq_lens)

    def arrange_targets(self, tensor_dict):
        slots, _, _ = tensor_dict["slots"]
        return slots

5. Output Layer
---------------

So far, our model is using the same components as any other model, including a common classification model, except for two things: the BiLSTMSlotAttention and the output layer.

BiLSTMSlotAttention is a multi-layer bidirectional LSTM based representation with attention over slots. The implementation of this representation is outside the scope of this tutorial, and this component is already included in PyText, so we'll just use it.

The output layer can be simple enough and demonstrates a few important notions in PyText, like how the loss function is tied to the output layer. We implement it like this:

.. code:: python

    class MyTaggingOutputLayer(OutputLayerBase):

        class Config(OutputLayerBase.Config):
            loss: CrossEntropyLoss.Config = CrossEntropyLoss.Config()

        @classmethod
        def from_config(cls, config, vocab, pad_token):
            return cls(
                vocab,
                create_loss(config.loss, ignore_index=pad_token),
            )

        def get_loss(self, logit, target, context, reduce=True):
            # flatten the logit from [batch_size, seq_lens, dim] to
            # [batch_size * seq_lens, dim]
            return self.loss_fn(logit.view(-1, logit.size()[-1]), target.view(-1), reduce)

        def get_pred(self, logit, *args, **kwargs):
            preds = torch.max(logit, 2)[1]
            scores = F.log_softmax(logit, 2)
            return preds, scores

6. Metric Reporter
------------------

Next we need to write a :class:`MetricReporter` to calculate metrics and report model training/test results:

The :class:`MetricReporter` base class aggregates all the output from Trainer, including predictions, scores and targets. The default aggregation behavior is concatenating the tensors from each batch and converting it to list. If you want different aggregation behavior, you can override it with your own implementation. Here we use the compute_classification_metrics method provided in pytext.metrics to get the precision/recall/F1 scores. PyText ships with a few common metric calculation methods, but you can easily incorporate other libraries, such as sklearn.

In the ``__init__`` method, we can pass a list of :class:`Channel` to report the results to any output stream. We use a simple :class:`ConsoleChannel` that prints everything to stdout and a :class:`TensorBoardChannel` that outputs metrics to TensorBoard:

.. code:: python

    class MyTaggingMetricReporter(MetricReporter):

        @classmethod
        def from_config(cls, config, vocab):
            return MyTaggingMetricReporter(
                channels=[ConsoleChannel(), TensorBoardChannel()],
                label_names=vocab
            )

        def __init__(self, label_names, channels):
            super().__init__(channels)
            self.label_names = label_names

        def calculate_metric(self):
            return compute_classification_metrics(
                list(
                    itertools.chain.from_iterable(
                        (
                            LabelPrediction(s, p, e)
                            for s, p, e in zip(scores, pred, expect)
                        )
                        for scores, pred, expect in zip(
                            self.all_scores, self.all_preds, self.all_targets
                        )
                    )
                ),
                self.label_names,
                self.calculate_loss(),
            )

7. Task
-------

Finally, we declare a task by inheriting from :class:`NewTask`. This base class specifies the training parameters of the model: the data source and batcher, the trainer class (most models will use the default one), and the metric reporter.

Since our metric reporter needs to be initialized with a specific vocab, we need to define the classmethod `create_metric_reporter` so that PyText can construct it properly.

.. code:: python

    class MyTaggingTask(NewTask):
        class Config(NewTask.Config):
            model: MyTaggingModel.Config = MyTaggingModel.Config()
            metric_reporter: MyTaggingMetricReporter.Config = MyTaggingMetricReporter.Config()

        @classmethod
        def create_metric_reporter(cls, config, tensorizers):
            return MyTaggingMetricReporter(
                channels=[ConsoleChannel(), TensorBoardChannel()],
                label_names=list(tensorizers["slots"].vocab),
            )

8. Generate sample config and train the model
---------------------------------------------

Save all your files in the same directory. For example, I saved all my files in ``my_tagging/``.Now you can tell PyText to include your classes with the parameter ``--include my_tagging``

Now that we have a fully functional class:`~Task`, we can generate a default JSON config for it by using the pytext cli tool.

.. code-block:: console

  (pytext) $ pytext --include my_tagging gen-default-config MyTaggingTask > my_config.json

Tweak the config as you like, for instance change the number of epochs. Most importantly, specify the path to your ATIS dataset. Then train the model with:

.. code-block:: console

  (pytext) $ pytext --include my_tagging train < my_config.json
