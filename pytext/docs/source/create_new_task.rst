Creating A New Task
======================================================

PyText uses a :class:`~Task` class as a central place to define components for data processing,
model training, metric reporting etc. and wire up those components. One can easily inherit from
an existing task and replace some (or all) components.

In this tutorial, we'll write a :class:`~WordTaggingTask`, and its associated components.

1. Define the :class:`~Task`
------------------------------

Usually features, targets, data_handler, model and metric_reporter are the components
subject to change, and we can reuse the other more general ones e.g trainer, optimizer and exporter ::

	from word_tagging import ModelInputConfig, TargetConfig

	class WordTaggingTask(Task):
	  class Config(Task.Config):
	    features: ModelInputConfig = ModelInputConfig()
	    targets: TargetConfig = TargetConfig()
	    data_handler: WordTaggingDataHandler.Config = WordTaggingDataHandler.Config()
	    model: WordTaggingModel.Config = WordTaggingModel.Config()
	    trainer: Trainer.Config = Trainer.Config()
	    optimizer: OptimizerParams = OptimizerParams()
	    scheduler: Optional[SchedulerParams] = SchedulerParams()
	    metric_reporter: WordTaggingMetricReporter.Config = WordTaggingMetricReporter.Config()
	    exporter: Optional[TextModelExporter.Config] = TextModelExporter.Config()

Every :class:`~Task` has an embedded :class:`~Config`, which defines the config of it's
components in a nested way. The base :class:`~Task` has a `from_config` method that creates every component and wires them up.

2. Define :class:`~ModelInput` and :class:`~Target`
----------------------------------------------------

The first two configs in the :class:`~Config` are model inputs (features) and targets
(expected outputs), which define the interface between data processing and model training. ::

	# word_tagging.py

	class ModelInputConfig(ModuleConfig):
	  word_feat: WordFeatConfig = WordFeatConfig()
	  dict_feat: Optional[DictFeatConfig] = None
	  char_feat: Optional[CharFeatConfig] = None

	class TargetConfig(ConfigBase):
	  # Transform sequence labels to BIO format
	  use_bio_labels: bool = False

:class:`~ModelInputConfig` defines all the possible input to our model, and will be used
in :class:`~DataHandler` to create TorchText :class:`~Field` to process raw data and also
in :class:`~Model` to create the first model layer: the **Embedding**.

3. Implement :class:`~DataHandler`
-----------------------------------

PyText uses the open source library `TorchText <https://github.com/pytorch/text>`_
for part of data preprocessing, including padding, numericalization and batching.
On top of TorchText, PyText incorporates a :class:`~Featurizer`, which provides data
processing steps that are shared in both training and inference time. Tokenization is
a typical step in Featurizer.

The general pipeline of a data handler is:

  1. Read data from a file into a list of raw data examples.
  2. Convert each row of row data to a TorchText :class:`~Example`.
  3. Generate a TorchText :class:`~Dataset` from the examples and a list of predefined TorchText :class:`~Field`
  4. Return a :class:`~BatchIterator` which will generate a tuple of (input, target, context) tensors for each iteration.

The base :class:`~DataHandler` already implements most of these steps, all we need to do is:

  1. Define the fields in `from_config` classmethod, a factory method to create a component from a config::

	@classmethod
	def from_config(cls, config: Config, model_input_config, target_config, **kwargs):
	    model_input_fields: Dict[str, Field] = create_fields(
	      model_input_config,
	        {
	            ModelInput.WORD_FEAT: TextFeatureField,
	            ModelInput.DICT_FEAT: DictFeatureField,
	            ModelInput.CHAR_FEAT: CharFeatureField,
	        },
	    )
	    target_fields: Dict[str, Field] = {WordLabelConfig._name: WordLabelField.from_config(target_config)}
	    extra_fields: Dict[str, Field] = {ExtraField.TOKEN_RANGE: RawField()}
	    kwargs.update(config.items())
	    return cls(
	        raw_columns=config.columns_to_read,
	        targets=target_fields,
	        features=model_input_fields,
	        extra_fields=extra_fields,
	        **kwargs,
	    )

We create input :class:`Field` by using the `create_fields` method which
combines the input config (first argument) with the provided map of name
to Class (second argument). Each :class:`Field` is constructed using its
`from_config` method with the matching config from the `input_config`.
Since this is a word labeling task, we need a :class:`Field` for the expected labels,
so we pass a single :class:`WordLabelField` into `target_fields` along with its column
name. Finally, we specify an extra field `token_range` which will be used later
to merge predicted word labels into the slots. Extra fields are processed but not
used directly by the model. They are passed along as batch context, which, as mentioned
above, will be used later in the process.

  2. Override the `preprocess_row` method to convert a row of raw data into a TorchText :class:`Example`::

	def preprocess_row(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
	      features = self.featurizer.featurize(
	          InputRecord(
	              raw_text=row_data.get(RawData.TEXT, ""),
	              raw_gazetteer_feats=row_data.get(RawData.DICT_FEAT, ""),
	          )
	      )
	      res = {
	          # features
	          ModelInput.WORD_FEAT: features.tokens,
	          ModelInput.DICT_FEAT: (
	              features.gazetteer_feats,
	              features.gazetteer_feat_weights,
	              features.gazetteer_feat_lengths,
	          ),
	          ModelInput.CHAR_FEAT: features.characters,
	          # target
	          [Target.WORD_LABEL_FIELD] = data_utils.align_slot_labels(
	              features.token_ranges,
	              row_data[RawData.WORD_LABEL],
	              self.targets[WordLabelConfig._name].use_bio_labels,
	          )
	          # extra data
	          BatchContext.TOKEN_RANGE: features.token_ranges,
	      }
	      return res

Here we invoke the :class:`Featurizer` and map the data to TorchText :class:`Field` names to
create a TorchText :class:`Dataset` later. Note the `data_utils.align_slot_labels`
method here, which breaks the slot labels that span multiple words into labels
for each word (with word labels and token ranges as inputs). We do the processing here because TorchText
assumes a 1:1 mapping between raw input and :class:`Field`.

4. Implement :class:`~Model`
------------------------------

A typical model in PyText is organized in four layers: **Embedding**, **Representation**,
**Decode** and **Output**. For any new model that conforms to this architecture,
writing the model is no more than just defining the config of each layer, since the
constructor and forward methods are already well defined in base :class:`~Model`.::

	class WordTaggingModel(Model):
	  class Config(ConfigBase):
	    representation: Union[BiLSTMSlotAttention.Config, BSeqCNNRepresentation.Config] = BiLSTMSlotAttention.Config()
	    decoder: MLPDecoder.Config = MLPDecoder.Config()
	    output_layer: Union[WordTaggingOutputLayer.Config, CRFOutputLayer.Config] = WordTaggingOutputLayer.Config()

You may notice that there's no config for the embedding layer here, because it
directly uses :class:`~ModelInputConfig`, already defined in the Task's :class:`~Config`.
By default, the embedding layer use :class:`~EmbeddingList` which creates a
list of sub embedding modules according to the :class:`~ModelInputConfig`, and concatenates their
vectors in the forward method. We don't need to override anything in
this example since the default behavior in base :class:`~Model` already does this::

	@classmethod
	def compose_embedding(cls, sub_embs):
	  return EmbeddingList(sub_embs.values(), concat=True)

the `sub_embs` parameter contains the embeddings we previously defined in the :class:`~ModelInputConfig`
(word_feat, dict_feat, char_feat).

If you're creating more complicated models, e.g PairNN, you can override this function
to reflect the embedding structure::

	@classmethod
	def compose_embedding(cls, sub_embs):
	  return EmbeddingList(
	    EmbeddingList(sub_embs["word_feat_1"], sub_embs["dict_feat_1"], concat=True),
	    EmbeddingList(sub_embs["word_feat_2"], sub_embs["dict_feat_2"], concat=True),
	    concat=False
	  )


Each layer can be either a single :class:`~Module` or a :class:`~Union` of multiple. In
this example, we give the user the choosing between two different types of representation
layers, which can be configured in config JSON file, with the default set to :class:`~BiLSTMSlotAttention`.

An example config of changing it to :class:`~BSeqCNNRepresentation` looks like::

	{
	  "model": {
	    "representation": {
	      "BSeqCNNRepresentation": {}
	    }
	  }
	}

The Decoder layer is a simple :class:`~MLPDecoder`.

The Output layer does three things -

  1) Computes loss
  2) Gets the prediction
  3) Exports to a Caffe2 net

Here we provide two options in this model: :class:`~WordTaggingOutputLayer` and :class:`~CRFOutputLayer`.
The former calculates a cross entropy loss and applies log softmax to get the prediction,
while the latter uses CRF (Conditional Random Fields) algorithm
to get both. The source code of both classes can be found in the PyText codebase. We'll
explain 3) in more detail in a following section.

**What if I have a completely different model structure?**
Then you can completely override both the `from_config` and `forward` methods in your
model class. However please inherit your model class from the base :class:`~Model` and use the
`create_module` method to construct modules. Doing so will give you the features of
freezing / saving / loading any part of the model for free. It's as easy as
setting the value in the corresponding config::
	{
	  "model": {
	    "representation": {
	      "BSeqCNNRepresentation": {
	        "freeze": true,
	        "save_path": "representation_layer.pt"
	        "load_path": "pretrained_representation_layer.pt"
	      }
	    }
	  }
	}


5. Implement :class:`~MetricReporter`
--------------------------------------

Next we need to write a :class:`~MetricReporter` to calculate metrics and report model training/test
results.::

	class WordTaggingMetricReporter(MetricReporter):
	    def __init__(self, channels, label_names, pad_index):
	        super().__init__(channels)
	        self.label_names = label_names
	        self.pad_index = pad_index

	    def calculate_metric(self):
	        return compute_classification_metrics(
	            list(
	                itertools.chain.from_iterable(
	                    (
	                        LabelPrediction(s, p, e)
	                        for s, p, e in zip(scores, pred, expect)
	                        if e != self.pad_index
	                    )
	                    for scores, pred, expect in zip(
	                        self.all_scores, self.all_preds, self.all_targets
	                    )
	                )
	            ),
	            self.label_names,
	        )

	    @staticmethod
	    def get_model_select_metric(metrics):
	        return metrics.accuracy

The :class:`~MetricReporter` base class already aggregates all the output from :class:`~Trainer`,
including predictions, scores and targets. The default aggregation behavior is
concatenating the tensors from each batch and converting it to list. If you
want different aggregation behavior, you can override it with your own
implementation. Here we use the `compute_classification_metrics` method provided in `pytext.metrics` to get the precision/recall/F1 scores.
PyText ships with a few common metric calculation methods, but you
can easily incorporate other libraries, such as sklearn.

Note that we also have to override the `get_model_select_metric` method to tell the
:class:`~Trainer`, how to select best model.

In the `__init__` method, we can pass a list of *Channel* to report
the results to any output stream. We use a simple :class:`~ConsoleChannel` that prints
everything to stdout and a :class:`~TensorBoardChannel` that outputs metrics to
`TensorBoard <https://www.tensorflow.org/guide/summaries_and_tensorboard>`_::

	class WordTaggingTask(Task):
	    # ... rest of the code
	    def create_metric_reporter(self):
	        return WordTaggingMetricReporter(
	            channels=[ConsoleChannel(), TensorBoardChannel()],
	            label_names=self.metadata.target.vocab.itos, # metadata is processed in DataHandler
	            pad_index=self.metadata.target.pad_index,
	        )

6. Implement the predict method
---------------------------------

With the code above, we can train and test the model. Next, we
need to add one more method in our :class:`~Trainer` to format the prediction results.
The base :class:`~Task` comes with a generic batch predict function that gets predictions
and scores from model and restores the order of input examples. By default it only returns
the raw numeric predictions, so we will override the `format_prediction` method and make it
more human readable::

	@classmethod
	def format_prediction(cls, predictions, scores, context, target_meta):
	    label_names = target_meta.vocab.itos
	    for prediction, score, token_ranges in zip(
	        predictions, scores, context[BatchContext.TOKEN_RANGE]
	    ):
	        yield [
	            {
	                "prediction": label_names[word_pred.data],
	                "score": {n: s for n, s in zip(label_names, word_score.tolist())},
	                "token_range": token_range,
	            }
	            for word_pred, word_score, token_range in zip(
	                prediction, score, token_ranges
	            )
	        ]

Note that we had created the `context[BatchContext.TOKEN_RANGE]` earlier as an extra field.

7. Implement :class:`~Exporter`
----------------------------------

The predict method is only used when experimenting with the model in PyTorch.
If we wish to run our model in the production-optimized Caffe2 environment, we'll have to create an :class:`~Exporter`.

An :class:`~Exporter` uses `ONNX <https://pytorch.org/docs/stable/onnx.html>`_ to
translate a PyTorch model to a Caffe2 net. After that, we prepend/append any additional
Caffe2 operators to the exported net. The default behavior in the base :class:`~Exporter` class
is to prepend a string-to-vector operator for vocabulary lookup and appending a operator
from model's output layer to format prediction results. In this exercise, that is all we
need, so we don't have to create a new :class:`~Exporter` here.

All that we need to do is implement the `export_to_caffe2` method in the output layer: ::

	class WordTaggingOutputLayer(OutputLayerBase):
	  def export_to_caffe2(
	      self, workspace, init_net, predict_net, model_out, output_name
	  ) -> List[core.BlobReference]:
	      scores = predict_net.Log(predict_net.Softmax(output_name, axis=2))
	      label_scores = predict_net.Split(scores, self.target_names, axis=2)
	      return [
	          predict_net.Copy(label_score, "{}:{}".format(output_name, name))
	          for name, label_score in zip(self.target_names, label_scores)
	      ]


8. Generate sample config and run the task
--------------------------------------------

Now that we have a fully functional class:`~Task`, we can generate a default JSON config for it by using the pytext cli tool

.. code-block:: console

	(pytext) $ pytext gen_default_config WordTaggingTask > task_config.json

Tweak the config as you like, and then train the model via

.. code-block:: console

	(pytext) $ pytext train < task_config.json

Run predictions using the trained PyTorch model

.. code-block:: console

	(pytext) $ pytext predict_py --model-file="YOUR_PY_MODEL_FILE" < test.json

Run predictions using the exported Caffe2 model

.. code-block:: console

	(pytext) $ pytext --config-file="task_config.json" predict --exported-model="YOUR_C2_MODEL_FILE" < test.json

Please refer to other tutorials in :doc:`index` for end to end working examples of training/predicting.
The full code of this example is also available in ``pytext.task``
