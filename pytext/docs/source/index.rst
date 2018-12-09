.. PyText documentation master file, created by
   sphinx-quickstart on Wed Oct  3 15:46:13 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/facebookresearch/pytext


PyText documentation
=====================

PyText is a deep-learning based NLP modeling framework built on PyTorch. PyText addresses the often-conflicting requirements of enabling rapid experimentation and of serving models at scale. It achieves this by providing simple and extensible interfaces and abstractions for model components, and by using PyTorch’s capabilities of exporting models for inference via the optimized Caffe2 execution engine. We are using PyText in Facebook to iterate quickly on new modeling ideas and then seamlessly ship them at scale.

**Core PyText features:**
- Production ready models for various NLP/NLU tasks:
  - Text classifiers
    - [Yoon Kim (2014): Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
    - [Lin et al. (2017): A Structured Self-attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)
  - Sequence taggers
    - [Lample et al. (2016): Neural Architectures for Named Entity Recognition](https://www.aclweb.org/anthology/N16-1030)
  - Joint intent-slot model
    - [Zhang et al. (2016): A Joint Model of Intent Determination and Slot Filling for Spoken Language Understanding](https://www.ijcai.org/Proceedings/16/Papers/425.pdf)
  - Contextual intent-slot models
- Distributed-training support built on the new C10d backend in PyTorch 1.0
- Extensible components that allows easy creation of new models and tasks
- Reference implementation and a pretrained model for the paper: [Gupta et al. (2018): Semantic Parsing for Task Oriented Dialog using Hierarchical Representations](http://aclweb.org/anthology/D18-1300)
- Ensemble training support

.. toctree::
  :maxdepth: 1
  :caption: Getting Started

  installation
  hack_on_pytext

.. toctree::
  :maxdepth: 1
  :caption: Tutorials

  hierarchical_intent_slot_tutorial
  atis_tutorial
  disjoint_multitask_tutorial
  distributed_training_tutorial
  create_new_task
  pytext_models_in_your_app

.. toctree::
  :maxdepth: 3
  :caption: Library Reference

  modules/pytext

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
