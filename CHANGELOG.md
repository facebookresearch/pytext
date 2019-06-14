# Changelog

## v0.2.0

*Note:* This release makes the new data handler API the default and deprecates Task and Model classes using the old data handler API. We recommend that you migrate your models to the new API as soon as possible. More details here: http://...

**New Stuff**
- most tasks and models deprecated, replaced with better versions using the new data handler API
- performance improvements in metric reporter
- Add Multilingual TSV Data Source
- LabelSmoothedCrossEntropyLoss
- Support for pretrained word embedding in TokenTensorizer
- option to use pretrained embedding
- TorchScript export for document classification
- Improve log in trainer
- performance measurement: reporting tokens_per_second and updates_per_second
- Implement DocumentReader from DrQA in PyText (StackedBidirectionalRNN)
- improved and updated documentation
- Implement SWA(SGD|ADAM) and Adagrad Optimizers
- cache numerized data in memory
- TorchScript BPE tokenization
- CLI command to update configs
- Visualize gradients with tensorboard

*Many bug fixes and code clean-ups*


## v0.1.5

*Note:* this is a last release in 0.1.x. The next release will deprecate Task and Model base classes and make the improved API of the new data handler the default. You can start using it already by inheriting from NewTask. NewDocumentClassification and NewWordTaggingTask use this new API, and you can get the first example in the tutorial "Custom Data Format".

**New Stuff**
- add config adapter
  - PyText is very young and its API is still in flux, making the config files brittle
  - config files now have a version number reflecting the API at the time it was created
  - older versions can be loaded and internally transformed into newer versions
- better metrics and reporting
  - better training time tracking
  - cool new visualization of model state in TensorBoard
  - pretty results in the terminal
- improved distributed training
- torchscript export
- support for SQuAD dataset
- add AugmentedLSTM
- add dense features support
- new plugin system: command line option --include to import custom user classes (see tutorial "Custom Data Format" for example)

*Many bug fixes and code clean-ups*


## v0.1.4
**New Stuff**
- Refactor Metric Reporters to reduce coupling
- RNNG Improvements:
  - Support Pretrained embeddings in RNNG
  - Support GPU Training
  - More Test Coverage
  - Tensorboard Support
- Added `QueryDocumentPairwiseRankingModel`
- Distributed Training Improvments:
  - Sharded Data Loading to reduce memory consumption
  - Fix Several issues with race conditions and unserializable state
- Reduced GPU memory Consumption by skipping gradient computation on evaluation

*And lots of bug fixes*

**Known Issues**
PyText doesn't work with the new ONNX v1.4.0, so we have pinned it to 1.3.0 for now


## v0.1.3
 - Remove epoch_size param from DisjointMultitask, use target_task (or shortest) to set epoch_size

## v0.1.0

Initial version
