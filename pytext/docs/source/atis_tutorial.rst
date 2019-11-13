Train Intent-Slot model on ATIS Dataset
======================================================

Intent detection and Slot filling are two common tasks in Natural Language Understanding for personal assistants. Given a user's "utterance" (e.g. Set an alarm for 10 pm), we detect its intent (set_alarm) and tag the slots required to fulfill the intent (10 pm).

The two tasks can be modeled as text classification and sequence labeling, respectively. We can train two separate models, but training a joint model has been shown to perform better.

1. Download dataset and Pre-trained embeddings
----------------------------------------------

In this tutorial, we will train a joint intent-slot model in PyText on the
`ATIS (Airline Travel Information System) dataset <https://www.kaggle.com/siddhadev/ms-cntk-atis/downloads/atis.zip/3>`_. Note that to download the dataset, you will need a `Kaggle <https://www.kaggle.com/>`_ account for which you can sign up for free.

Download the data locally and unzip it.

.. code-block:: console

    $ unzip <download_dir>/atis.zip -d atis

Word embeddings are the vector representations of the different words understood by your model. Pre-trained word embeddings can significantly improve the accuracy of your model, since they have been trained on vast amounts of data. In this tutorial, we'll use `GloVe embeddings <https://nlp.stanford.edu/projects/glove/>`_, which can be downloaded by:

.. code-block:: console

    $ curl https://nlp.stanford.edu/data/wordvecs/glove.6B.zip > glove.6B.zip
    $ unzip glove.6B.zip -d atis

The downloaded file size is ~800 MB.

2. Use Custom DataSource
------------------------

In many cases, datasets are in TSV format and can use :class:`~TSVDataSource`. However, ATIS dataset is different and needs to be pre-processed or use a custom :class:`~DataSource`.

The format of the ATIS dataset is discussed in details in datasource_tutorial. This tutorial gives the source code that loads the queries and intent, and we need to add the slots. This is not straightforward, because the ATIS data uses BIO annotations, and our model uses :class:`~SlotLabelTensorizer`, which expects the slots to be described with `<start1>:<end1>:<label1>,<start2>:<end2>:<label2>,...`.

The source code will not be detailed here, but it can be found in `demo/datasource/atis_intent_slot.py`.

3. Train the model
--------------------------

To train a PyText model, you need to pick the task and model architecture, among other parameters. Default values are available for many parameters and can give reasonable results in most cases. The example config in `demo/configs/atis_intent_slot.json` can train a joint intent-slot model ::

    {
        "include_dirs": ["demo/datasource"],
        "export_caffe2_path": "atis_intent_slot.c2",
        "task": {
            "IntentSlotTask": {
                "data": {
                    "Data": {
                        "source": {
                            "AtisIntentSlotsDataSource": {
                                "path": "atis"
                            }
                        },
                        "batcher": {
                            "PoolingBatcher": {
                                "train_batch_size": 128,
                                "eval_batch_size": 128,
                                "test_batch_size": 128
                            }
                        },
                        "sort_key": "tokens"
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
                        "pretrained_embeddings_path": "atis/glove.6B.100d.txt"
                    }
                },
                "trainer": {
                    "epochs": 20
                }
            }
        },
        "version": 17
    }

We explain some of the parameters involved:

- :class:`~IntentSlotTask` trains a joint model for document classification and word tagging.
- :class:`~AtisIntentSlotsDataSource` is our custom data source that can read the ATIS dataset directly.
- The :class:`~Model` has multiple layers -
  - We use BiLSTM model with attention as the representation layer. The pooling attribute decides the attention technique used.
  - We use different loss functions for document classification (Cross Entropy Loss) and slot filling (CRF layer)
- Pre-trained word embeddings are provided within the `word_embedding` attribute.

To train the PyText model, simply run the train command:

.. code-block:: console

    (pytext) $ pytext train < demo/configs/atis_intent_slot.json


3. Tune the model and get final results
-----------------------------------------

Tuning the model's hyper-parameters is key to obtaining the best model accuracy. Using hyper-parameter sweeps on learning rate, number of layers, dimension and dropout of BiLSTM etc., we can achieve a F1 score of ~95% on slot labels which is close to the state-of-the-art. The fine-tuned model config is available at `demo/configs/atis_intent_slot2.json`

To train the model using fine tuned model config,

.. code-block:: console

    (pytext) $ pytext train < demo/configs/atis_intent_slot2.json

4. Generate predictions
-----------------------------------------

With the model trained, we can export a caffe2 model. (We first need to install tensoboard for some reason.)

.. code-block:: console

    (pytext) $ pip install tensorboard
    (pytext) $ pytext --config-file demo/configs/atis_intent_slot.json export

Lets make the model run on some sample utterances! You can input one by running:

.. code-block:: console

    (pytext) $ pytext --config-file demo/configs/atis_intent_slot.json \
        predict <<< '{"text": "flights from colorado"}'

The response from the model is log of probabilities for different intents and slots, with the correct intent and slot hopefully having the highest. In the following snippet of the model's response, we see that the intent `doc_scores:flight` and slot `word_scores:fromloc.city_name` for third word "colorado" have the highest predictions.

.. code-block:: console

    {
     ....
     'doc_scores:city': array([-7.9052157], dtype=float32),
     'doc_scores:distance': array([-6.7239347], dtype=float32),
     'doc_scores:flight': array([-0.05219311], dtype=float32),
     'doc_scores:flight+airfare': array([-5.8061013], dtype=float32),
     'doc_scores:flight_no': array([-6.832335], dtype=float32),
     'doc_scores:flight_time': array([-4.6331463], dtype=float32),
     ..,
     'word_scores:fromloc.airport_name': array([[-7.965843 ],
           [-6.498479 ],
           [-1.2838008]], dtype=float32),
     'word_scores:fromloc.city_name': array([[-8.286056 ],
           [-6.8367434],
           [-1.7030029]], dtype=float32),
     'word_scores:fromloc.state_code': array([[-11.213549],
           [-10.83372 ],
           [ -5.261614]], dtype=float32),
     'word_scores:fromloc.state_name': array([[-9.108554],
     ...
    }
