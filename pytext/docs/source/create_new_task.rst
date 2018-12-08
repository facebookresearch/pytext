Create A New Task
======================================================

PyText uses a Task class as a central place to define components for data processing,
model training, metric reporting, etc, and wire up these components. It's very easy
to replace some components in existing Task classes by just inheriting from the Task
and drop in your own components. Let's use WordTagging Task as an example to demonstrate
how to create a new task and make it end to end work.

1. Define the task class
-------------------------

First let's write the WordTaggingTask class, usually features, targets, data_handler,
model and metric_reporter are the parts subject to change, and we can reuse the other
general components, like the trainer, optimizer, exporter::

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

Every Task class has a embedded Config class, which defines the config of it's
components in a nested way. The base Task class has a from_config method that will
create every components and wire them up.

2. Define ModelInput and Target
--------------------------

The first two configs in the Config class are model inputs(features) and targets
(expected outputs), which basically define the interface between data processing
and model training.::
	# word_tagging.py file

	class ModelInputConfig(ModuleConfig):
	  word_feat: WordFeatConfig = WordFeatConfig()
	  dict_feat: Optional[DictFeatConfig] = None
	  char_feat: Optional[CharFeatConfig] = None

	class TargetConfig(ConfigBase):
	  # Transform sequence labels to BIO format
	  use_bio_labels: bool = False

The ModelInputConfig defines all the possible input to our model, and will be used
in DataHandler class to create Field to process raw data and also in Model class to
create the first model layer: the **Embedding** layer

3. Write DataHandler
--------------------------
PyText is using the open source library `TorchText <https://github.com/pytorch/text>`_
for part of data preprocessing, including padding, numericalization and batching.
On top of TorchText, PyText incorporates a Featurizer concept, which provides data
process steps that are shared in both training and inference time. Tokenization is
a typical step in Featurizer. So the general pipeline of data handler is:

  1. Read data from file into a list of raw data examples
  2. Convert each row of row data to a TorchText Example.
  3. Generate a TorchText.Dataset by using the list of Example from step 2 and a
  list of predefined TorchText.Field
  4 Return a BatchIterator which will generate a tuple of (input, target, context)
  tensors for each iteration.

The base DataHandler class already covers most of the content of these steps, what
we have to do is:

  1. Define the fields in from_config class method of our sub class, from_config
  is a factory method to create component using config::

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

  We created several feature Fields by using the create_fields function which automatically
  aligns Field class/config and creates Field using it's from_config function. Also
  created a single WordLabelField and an extra field token_range. Extra fields will
  process and pass along data as batch context, which will not directly used by model,
  in this case it will be used later to merge predicted word labels into slots.

  2. Override the preprocess_row row function to convert a row of raw data to TorchText.Example::

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

  It basically invokes Featurizer and maps the data to TorchText Field names to
  create TorchText Dataset later. Please notice the ``data_utils.align_slot_labels``
  function here, it breaks the slot label that spans multiple words into labels
  for each word, the function requires two inputs, word labels and token ranges.
  We're doing the processing here instead of in TorchText.Field because TorchText
  assumes a 1:1 mapping between raw input and Field.

4. Write Model
--------------------------

A typical model in PyText is organized in four layers: **Embedding** Layer, **Representation**
layer, **Decode** layer and **Output** layer. For any new model that conforms to this architecture,
writing the model is no more than just define the config of each layer since the
construction and forward functions are already well defined in base Model.::

	class WordTaggingModel(Model):
	  class Config(ConfigBase):
	    representation: Union[
	      BiLSTMSlotAttention.Config, BSeqCNNRepresentation.Config
	    ] = BiLSTMSlotAttention.Config()
	    decoder: MLPDecoder.Config = MLPDecoder.Config()
	    output_layer: Union[
	      WordTaggingOutputLayer.Config, CRFOutputLayer.Config
	    ] = WordTaggingOutputLayer.Config()

You may notice there's no config for embedding layer here, it's because embedding
layer uses ModelInputConfig as it's config, which is already defined in the Task
Config class. By default, embedding layer use EmbeddingList class which creates a
list of sub embedding modules according to the ModelInputConfig, and concat the
embedding vectors of them in forward function. We don't have to override anything in
this example since the default behavior in base Model class already did this::

	@classmethod
	def compose_embedding(cls, sub_embs):
	  return EmbeddingList(sub_embs.values(), concat=True)

the sub_embs parameter contains the embeddings we previously defined in the ModelInputConfig
(word_feat, dict_feat, char_feat).

if you're creating more complicated models, e.g pairNN, you can override this function
to reflect the embedding structure::

	@classmethod
	def compose_embedding(cls, sub_embs):
	  return EmbeddingList(
	    EmbeddingList(sub_embs["word_feat_1"], sub_embs["dict_feat_1"], concat=True),
	    EmbeddingList(sub_embs["word_feat_2"], sub_embs["dict_feat_2"], concat=True),
	    concat=False
	  )


Each layer can be either a single Module class or a Union of different Module. In
this example, we give the user the option of two different approaches to the representation
layer, which can be configured in config json file, by default it's BiLSTMSlotAttention,
if not specified in json.
An example config of changing it to BSeqCNNRepresentation looks like::

	{
	  "model": {
	    "representation": {
	      "BSeqCNNRepresentation": {}
	    }
	  }
	}

Decoder layer is just a simple MLPDecoder.

Output layer is a special layer that do three things:
  1) compute loss
  2) get prediction
  3) export to caffe2 model

Here we provide two options in this model: WordTaggingOutputLayer and CRFOutputLayer.
WordTaggingOutputLayer calculates a cross entropy loss and applies log softmax to
get prediction, while CRFOutputLayer uses CRF(Conditional Random Fields) algorithm
to get both. The source code of both classes can be found in PyText codebase. We'll
explain more about 3) in following section.

**What if I have a completely different model structure?**
Then you can completely override both the from_config and forward function in your
model class. However please inherit your model class from base Model class and use
create_module function to construct modules in your model, by doing that you can
get the feature of freeze/save/load any part of the model for free. It's as easy as
setting the value if the corresponding config::
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


5. Write MetricReporter
--------------------------

Next we need to write a MetricReporter to calculate metrics and report model training/test
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

MetricReporter base class already aggregates all the output from Trainer, including
predictions, scores, targets. The default aggregation behavior is concatenating
the tensors from each batch and converting it to list, you can override it if you
want to aggregate in a different way. To compute the metrics, here we use a
``compute_classification_metrics`` function provided in ``pytext.metrics`` Module
to get the precision/recall/f1 score. PyText is shipped with a bunch of common metric
calculation methods, but you can always use methods from other open source libraries,
such as sklearn.

Notice we also have to override the ``get_model_select_metric`` method to tell
Trainer how to select best model.

In the construction function, we can pass a list of *Channel* to report the results
to any output stream. We use a simple ConsoleChannel that prints everything to
stdout and a TensorBoardChannel that output metrics to TensorBoard for our task::

	class WordTaggingTask(Task):
	    # ... rest of the code
	    def create_metric_reporter(self):
	        return WordTaggingMetricReporter(
	            channels=[ConsoleChannel(), TensorBoardChannel()],
	            label_names=self.metadata.target.vocab.itos, # metadata is processed in DataHandler
	            pad_index=self.metadata.target.pad_index,
	        )

6. Write Exporter
--------------------------

Content goes here. Content goes here. Content goes here. Content goes here.

7. Generate sample config and run the task
--------------------------

TBD

8 Write predict function
--------------------------

TBD
