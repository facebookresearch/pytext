Train Intent-Slot model on ATIS Dataset
======================================================

**OBSOLETE** This documentation is using the old API and needs to be updated with the new classes configs.

Intent detection and Slot filling are two common tasks in Natural Language Understanding for personal assistants. Given a user's "utterance" (e.g. Set an alarm for 10 pm), we detect its intent (set_alarm) and tag the slots required to fulfill the intent (10 pm).

The two tasks can be modeled as text classification and sequence labeling, respectively. We can train two separate models, but training a joint model has been shown to perform better.

In this tutorial, we will train a joint intent-slot model in PyText on the
`ATIS (Airline Travel Information System) dataset <https://www.kaggle.com/siddhadev/ms-cntk-atis/downloads/atis.zip/3>`_. Note that to download the dataset, you will need a `Kaggle <https://www.kaggle.com/>`_ account for which you can sign up for free.


1. Prepare the data
-------------------------

The in-built PyText data-handler expects the data to be stored in a tab-separated file that contains the intent label, slot label and the raw utterance.

Download the data locally and use the script below to preprocess it into format PyText expects

.. code-block:: console

    $ unzip <download_dir>/atis.zip -d <download_dir>/atis
    $ python3 demo/atis_joint_model/data_processor.py
      --download-folder <download_dir>/atis --output-directory demo/atis_joint_model/

The script will also randomly split the training data into training and validation sets. All the pre-processed data will be written to the output-directory argument specified in the command.

An alternative approach here would be to write a custom data-handler for your custom data format, but that is beyond the scope of this tutorial.

2. Download Pre-trained word embeddings
---------------------------------------------

Word embeddings are the vector representations of the different words understood by your model. Pre-trained word embeddings can significantly improve the accuracy of your model, since they have been trained on vast amounts of data. In this tutorial, we'll use `GloVe embeddings <https://nlp.stanford.edu/projects/glove/>`_, which can be downloaded by:

.. code-block:: console

    $ curl https://nlp.stanford.edu/data/wordvecs/glove.6B.zip > demo/atis_joint_model/glove.6B.zip
    $ unzip demo/atis_joint_model/glove.6B.zip -d demo/atis_joint_model

The downloaded file size is ~800 MB.

3. Train the model
--------------------------

To train a PyText model, you need to pick the right task and model architecture, among other parameters. Default values are available for many parameters and can give reasonable results in most cases. The following is a sample config which can train a joint intent-slot model ::

    {
      "config": {
        "task": {
          "IntentSlotTask": {
            "data": {
              "Data": {
                "source": {
                  "TSVDataSource": {
                    "field_names": [
                      "label",
                      "slots",
                      "text",
                      "doc_weight",
                      "word_weight"
                    ],
                    "train_filename": "demo/atis_joint_model/atis.processed.train.csv",
                    "eval_filename": "demo/atis_joint_model/atis.processed.val.csv",
                    "test_filename": "demo/atis_joint_model/atis.processed.test.csv"
                  }
                },
                "batcher": {
                  "PoolingBatcher": {
                    "train_batch_size": 128,
                    "eval_batch_size": 128,
                    "test_batch_size": 128,
                    "pool_num_batches": 10000
                  }
                },
                "sort_key": "tokens",
                "in_memory": true
              }
            },
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
              },
              "word_embedding": {
                "embed_dim": 100,
                "pretrained_embeddings_path": "demo/atis_joint_model/glove.6B.100d.txt"
              }
            },
            "trainer": {
              "epochs": 20,
              "optimizer": {
                "Adam": {
                  "lr": 0.001
                }
              }
            }
          }
        }
      }
    }

We explain some of the parameters involved:

- :class:`~IntentSlotTask` trains a joint model for document classification and word tagging.
- The :class:`~Model` has multiple layers -
  - We use BiLSTM model with attention as the representation layer. The pooling attribute decides the attention technique used.
  - We use different loss functions for document classification (Cross Entropy Loss) and slot filling (CRF layer)
- Pre-trained word embeddings are provided within the `word_embedding` attribute.

To train the PyText model,

.. code-block:: console

    (pytext) $ pytext train < sample_config.json


3. Tune the model and get final results
-----------------------------------------

Tuning the model's hyper-parameters is key to obtaining the best model accuracy. Using hyper-parameter sweeps on learning rate, number of layers, dimension and dropout of BiLSTM etc., we can achieve a F1 score of ~95% on slot labels which is close to the state-of-the-art. The fine-tuned model config is available at ``demos/atis_intent_slot/atis_joint_config.json``

To train the model using fine tuned model config,

.. code-block:: console

    (pytext) $ pytext train < demo/atis_joint_model/atis_joint_config.json


4. Generate predictions
-----------------------------------------

Lets make the model run on some sample utterances! You can input one by running

.. code-block:: console

    (pytext) $ pytext --config-file demo/atis_joint_model/atis_joint_config.json \
      predict --exported-model /tmp/atis_joint_model.c2 <<< '{"text": "flights from colorado"}'

The response from the model is log of probabilities for different intents and slots, with the correct intent and slot hopefully having the highest.

In the following snippet of the model's response, we see that the intent `doc_scores:flight` and slot `word_scores:fromloc.city_name` for third word "colorado" have the highest predictions. ::

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
