Training Joint Intent-Slot on ATIS Dataset
======================================================

Intent detection and Slot filling are two common tasks in Natural Language Understanding. In intent detection, we try to find the objective that the utterance is trying to achieve. Get Directions and Set Alarm are two examples. In slot filling, we assign words different labels based on their purpose. For example, if the intent is Get Direction, Source_Location and Target_Location can be considered as two valid slots. While intent detection is a semantic classification task, slot filling is a sequence labeling task, where you assign labels to different tokens in the utterance. These two models can be either trained separately or they can be trained together. Training a joint model has been generally preferred over two different models.

In this tutorial, we will train an intent-slot model in PyText using the
`ATIS (Airline Travel Information System) dataset <https://www.kaggle.com/siddhadev/ms-cntk-atis/downloads/atis.zip/3>`_. Note that to download the dataset, you will need a `Kaggle <https://www.kaggle.com/>`_ account which you can signup for free.


1. Preparing the data.
-------------------------

The builtin PyText data-handler expects the data to be stored in a tab-separated file that
contains the intent label, slot label and the raw utterance.
The first step is to download the data locally and you can use the below
script to preprocess the data to the PyText format::

    > python3 demo/atis_joint_model/data_processor.py
      --download-folder ./download_dir --output-directory demo/atis_joint_model/

The script will also randomly split the train data into train and validation. All the pre-processed data will be written to output-directory parameter that is specified in the command.

Another alternative approach here is to write a custom data-handler for your custom data format. We won't be using this method in this tutorial.

2. Download Pre-trained word embeddings
---------------------------------------------

Pre-trained word embeddings can help improve accuracy of your model because they are trained on vast amounts of data. In this tutorial, we will be using `GloVe embeddings <https://nlp.stanford.edu/projects/glove/>`_.
GloVe embeddings can be downloaded locally using below commands. Downloaded file size is approximately 800MB::

    > wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip -P demo/atis_joint_model/
    > unzip demo/atis_joint_model/glove.6B.zip -d demo/atis_joint_model

These pre-trained word embeddings are used in the model to get the vector representation for different tokens in the utterances.

3. Training the model.
--------------------------

To train a PyText model, you need to pick the right task and model architecture
among other parameters. Default values are available for many parameters and can
give reasonable results in most cases. Below is a simple config which can help
you train a joint intent-slot model::

    {
      "config": {
        "task": {
          "JointTextTask": {
            "model": {
              "representation": {
                "BiLSTMDocSlotAttention": {
                  "pooling": {
                    "SelfAttention": {}
                  }
                }
              },
              "output_layer": {
                "doc_output": {
                  "loss": {
                    "CrossEntropyLoss": {}
                  }
                },
                "word_output": {
                  "CRFOutputLayer": {}
                }
              }
            },
            "features": {
              "word_feat": {
                "embed_dim": 100,
                "pretrained_embeddings_path": "demo/atis_joint_model/glove.6B.100d.txt",
              }
            },
            "optimizer": {
              "type": "adam",
              "lr": "0.001"
            },
            "trainer": {
              "epochs": 20
            },
            "featurizer": {
              "SimpleFeaturizer": {}
            },
            "data_handler": {
              "train_path": "demos/atis_joint_model/atis.processed.train.csv",
              "eval_path": "demos/atis_joint_model/atis.processed.val.csv",
              "test_path": "demos/atis_joint_model/atis.processed.test.csv"
            }
          }
        }
      }
    }


Explanation for some parameters can be found below

- JointTextTask is used to train the joint model for document classification and word tagging.
- For the representation layer, we use a BiLSTM model. It has options to enable attention for both intent classification as well as slot filling.

    - pooling attribute decides the attention technique used for document classification.
- For the output layer, we use different loss functions for document classification and slot filling

    - CrossEntropyLoss is used for intent detection.
    - CRF layer on top of different slot probabilities is used for the slot filling task.
- Pre-trained word embeddings can be provided using `word_feat` attribute inside `features`.
- For featurizer, we use SimpleFeaturizer to do space based tokenization of the utterance.


To train the PyText model::

    > pytext train < sample_config.json

Note that config referenced in the next section can help you train a model which gives you an accuracy very close to the SOTA model.

3. Model tuning and final results.
-----------------------------------------

Tuning the model parameters is key to obtaining the best model accuracy. Using parameter sweep on different parameters like learning rate and, number of layers, dimension and dropout of BiLSTM,
we can achieve an F1 score of around 95% on slot labels which is very close to the SOTA F1 score. Fined tuned model config is available at ``demos/atis_intent_slot/atis_joint_config.json``

To train the model using fine tuned model config::

    > pytext train < demo/atis_joint_model/atis_joint_config.json


4. Generating predictions.
-----------------------------------------

Once you have trained and exported your model, it is very easy to make predictions using the model. You need to pass your utterance as a json to the model ::

    > pytext --config-file demo/atis_joint_model/atis_joint_config.json \
      predict --exported-model /tmp/atis_joint_model.c2 <<< '{"raw_text": "flights from colorado"}'

Response from the model will be log of probabilities for different intents and slots. Below is part of the model response. It can be seen that correct intent and slot has the highest probability.

It can be seen that intent `doc_scores:flight` and slot `word_scores:fromloc.city_name` for third word have the highest predictions.::

    {
     ....
     'doc_scores:flight': array([-0.00016726], dtype=float32),
     'doc_scores:ground_service+ground_fare': array([-25.865768], dtype=float32),
     'doc_scores:meal': array([-17.864975], dtype=float32),
     ..,
     'word_scores:airline_name': array([[-12.158762],
           [-15.142928],
           [ -8.991585]], dtype=float32),
     'word_scores:fromloc.city_name': array([[-1.5084317e+01],
           [-1.3880151e+01],
           [-1.4416825e-02]], dtype=float32),
     'word_scores:fromloc.state_code': array([[-17.824356],
           [-17.89767 ],
           [ -9.848984]], dtype=float32),
     'word_scores:meal': array([[-15.079164],
           [-17.229427],
           [-17.529446]], dtype=float32),
     'word_scores:transport_type': array([[-14.722928],
           [-16.700478],
           [-13.4414  ]], dtype=float32),
     ...
    }
