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
TBD

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

Content goes here. Content goes here. Content goes here. Content goes here.

6. Write Exporter
--------------------------

Content goes here. Content goes here. Content goes here. Content goes here.

7. Generate sample config and run the task
--------------------------


