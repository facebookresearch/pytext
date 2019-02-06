# Changelog

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
