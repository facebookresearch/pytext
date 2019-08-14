# Changelog

## v0.2.2

*Note:* this is the last release with _Deprecated classes. Those classes will be removed in the next release.

**New Features:**
- DeepCNN Representation for word tagging
- Combine KLDivergenceBCELoss with SoftHardBCELoss and F.cross_entropy() in CrossEntropyLoss (#689)
- add dense feature support for doc model (#710)
- add torchscript quantizaiton support in pytext
- pytext multi-label support (#731)
- open source transformer representations (#736)
- open source transformer based models - data, tensorizers and tokenizer (#708)
- Create AlternatingRandomizedBatchSampler (#737)
- open source MaskedLM and BERT models (#734)
- Support bytes input in word tagging model OSS (#745)
- open source extractive question answering models (#742)
- torchscriptify for ensemle task
- enabled lmlstm labels exporting (#767)
- Enable dense features in ByteTokensDocumentModel (#763)
- created bilstm dropout condition (#769)
- enabled lmlstm caffe2 exporting (#766)
- PolynomialDecayScheduler (#791)
- removed bilstm dependence on seq_lengths (#776)
- fp16 optimizer (#782)
- Add Dense Feature Normalization to FloatListTensorizer and DocModel (#859)
- Add Sparsifier component to PyText and L0-projection based sparsifier (#860)
- implemented cnn pooling for doc classification (#872)
- implemented bottleneck separable convolutions (#855)
- Add eps to Adam (#858)
- implemented mobile exporter (#785)
- support starting training from saved checkpoint (#824)
- implemented separable convolutions (#830)
- implemented gelu activations (#829)
- implemented causal convolutions (#811)
- implemented dilation for convolutions (#810)
- created weight norm option (#809)
- Ordered Neuron LSTM (#854)
- Add PersonalizedByteDocModel (#816)
- CNN based language models (#827)
- improve csv support in TSVDataSource (#777)
- Change default batch sampler DisjointMultitaskData to RoundRobinBatchSampler (#802)
- Support using serialized pretrained embedding file (#797)

**Documentation / Usability / Logging:**
- Fewer out-of-vocab print messages, with some stats (#697)
- Echo epoch number to console while training (#712)
- Separate timing for prediction and metric calculation. (#738)
- multi-label soft metrics (#754)
- changed lm metric reporting (#765)
- fix data source tutorial (#762)
- fix doc sphinx deprecation warning (#775)
- Add the ability to pass parameter values to gen-default-config (#856)
- Remove "pytext/" from paths in demo json config (#878)
- New documentation about hacking pytext and dealing with github. (#862)
- install_deps supports updates (#863)
- Reduce number of PEP print (#861)
- better error message for config with unknown component (#801)
- Add Recall at Precision Thresholds to Config (#792)
- implemented perplexity reductions for lm score reporting (#799)
- adapt prediction workflow to new design (#746)

**Bug fixes:**
- block sharded tsv eval/test fix (#698)
- Fix BoundaryPooling tracing (#713)
- fixes LMLSTM weight tying bug (#704)
- Fix duplicate entries in vocab (#721)
- Bugfix for trainer not reporting eval results (#740)
- Reintroduce metrics export in new task (#748)
- fix open source tests (#750)
- Fix missing init_tensorizers arg (#893)
- Fix intent slot metric reporter not working with byte offset (#883)
- Fix issue with some tensorizers still re-initializing vocab when loaded from saved state (#848)
- fixed overflow error in lm reporting (#831)
- fix BlockShardedTSVDataSource (#832)


## v0.2.1

(skipped because of packaging issues)


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
