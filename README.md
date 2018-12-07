# Overview

[![CircleCI](https://circleci.com/gh/facebookresearch/pytext.svg?style=svg&circle-token=2e0e0cb6dc686b646df887c2e0f07a8429712243)](https://circleci.com/gh/facebookresearch/pytext)

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

# Installing PyText

To get started, run the following commands in a terminal:

```
git clone git@github.com:facebookresearch/pytext.git
cd pytext

source activation_venv
./install_deps
```
[Install PyTorch Using Pip](https://pytorch.org/) make sure to get the correct version for your OS and GPU Situation.
Once that is installed, you can run the unit tests with:
```
./run_tests
```

To resume development in an already checked-out repo:

```
cd pytext
source activation_venv
```

To exit the virtual environment:

```
deactivate
```

Alternatively, if you don't want to run in a virtual env, you can install the dependencies globally with `sudo ./install_deps`.

For additional information, please read INSTALL.md

# Train your first text classifier

For this first example, we'll train a CNN-based text-classifier that classifies text utterances, using the examples in `tests/data/train_data_tiny.tsv`.

```
python3 pytext/main.py train < demo/configs/docnn.json
```

By default, the model is created in `/tmp/model.pt`

Now you can export your model as a caffe2 net:

```
pytext export < config.json
```

You can use the exported caffe2 model to predict the class of raw utterances like this:

```
pytext --config-file config.json predict <<< '{"raw_text": "create an alarm for 1:30 pm"}'
```

# License
PyText is BSD-licensed, as found in the LICENSE file.
