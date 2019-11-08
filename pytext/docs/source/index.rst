.. PyText documentation master file, created by
   sphinx-quickstart on Wed Oct  3 15:46:13 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/facebookresearch/pytext


PyText Documentation
=====================

PyText is a deep-learning based NLP modeling framework built on PyTorch. PyText addresses the often-conflicting requirements of enabling rapid experimentation and of serving models at scale. It achieves this by providing simple and extensible interfaces and abstractions for model components, and by using PyTorch’s capabilities of exporting models for inference via the optimized Caffe2 execution engine. We use PyText at Facebook to iterate quickly on new modeling ideas and then seamlessly ship them at scale.

**Core PyText Features:**

- Production ready models for various NLP/NLU tasks:

  - Text classifiers

    - `Yoon Kim (2014): Convolutional Neural Networks for Sentence Classification <https://arxiv.org/abs/1408.5882>`_
    - `Lin et al. (2017): A Structured Self-attentive Sentence Embedding <https://arxiv.org/abs/1703.03130>`_

  - Sequence taggers

    - `Lample et al. (2016): Neural Architectures for Named Entity Recognition <https://www.aclweb.org/anthology/N16-1030>`_

  - Joint intent-slot model

    - `Zhang et al. (2016): A Joint Model of Intent Determination and Slot Filling for Spoken Language Understanding <https://www.ijcai.org/Proceedings/16/Papers/425.pdf>`_

  - Contextual intent-slot models

- Extensible components that allow easy creation of new models and tasks
- Ensemble training support
- Distributed-training support (using the new C10d backend in PyTorch 1.0)
- Reference implementation and a pre-trained model for the paper: `Gupta et al. (2018): Semantic Parsing for Task Oriented Dialog using Hierarchical Representations <http://aclweb.org/anthology/D18-1300>`_

How To Use
============

Please follow the tutorial series in **Getting Started** to get a sense of how to train a basic model and deploy to production.

After that, you can explore more options of builtin models and training methods in **Training More Advanced Models**

If you want to use PyText as a library and build your own models, please check the tutorial in **Extending PyText**

.. note::
  All the demo configs and test data for the tutorials can be found in source code. You can either install PyText from source or download the files manually from GitHub.

.. toctree::
  :maxdepth: 1
  :caption: Getting Started

  installation
  train_your_first_model
  execute_your_first_model
  visualize_your_model

  pytext_models_in_your_app
  serving_models_in_production

  config_files
  config_commands

.. toctree::
  :maxdepth: 1
  :caption: Training More Advanced Models

  atis_tutorial
  hierarchical_intent_slot_tutorial
  disjoint_multitask_tutorial
  distributed_training_tutorial
  xlm_r

.. toctree::
  :maxdepth: 1
  :caption: Extending PyText

  overview
  datasource_tutorial
  tensorizer
  dense
  create_new_model
  hacking_pytext

.. toctree::
  :maxdepth: 2
  :caption: References

  configs/pytext
  modules/pytext

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
