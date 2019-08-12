Train your first model
==================================

Once you've `installed PyText <installation.html>`_ you can start training your first model!

This tutorial series is an overview of *using* PyText, and will cover the main concepts PyText uses to interact with the world. It won't deal with modifying the code (e.g. hacking on new model architectures). By the end, you should have a high-quality text classification model that can be used in production.

You can use PyText as a library either in your own scripts or in a Jupyter notebook, but the fastest way to start training is through the PyText command line tool. This tool will automatically be in your path when you install PyText!

.. code-block:: console

  (pytext) $ pytext

  Usage: pytext [OPTIONS] COMMAND [ARGS]...

    Configs can be passed by file or directly from json. If neither --config-
    file or --config-json is passed, attempts to read the file from stdin.

    Example:

      pytext train < demos/docnn.json

  Options:
    --config-file TEXT
    --config-json TEXT
    --help              Show this message and exit.

  Commands:
    export   Convert a pytext model snapshot to a caffe2 model.
    predict  Start a repl executing examples against a caffe2 model.
    test     Test a trained model snapshot.
    train    Train a model and save the best snapshot.

Background
----------

Fundamentally, "machine learning" means learning a function automatically. Your training, evaluation, and test datasets are examples of inputs and their corresponding outputs which show how that function behaves. A **model** is an implementation of that function. To **train** a **model** means to make a statistical implementation of that function that uses the training data as a rubric. To **predict** using a **model** means to take a trained implementation and apply it to new inputs, thus predicting what the result of the idealized function would be on those inputs.

More examples to train on usually corresponds to more accurate and better-generalizing models. This can mean thousands to millions or billions of examples depending on the task (function) you're trying to learn.

PyText Configs
---------------

Training a state-of-the-art PyText model on a dataset is primarily about configuration. Picking your training dataset, your model parameters, your training parameters, and so on, is a central part of building high-quality text models.

Configuration is a central part of every component within PyText, and the config system that we provide allows for all of these configurations to be easily expressible in JSON format. PyText comes in-built with a number of example configurations that can train in-built models, and we have a system for automatically documenting the default configurations and possible configuration values.

PyText Modes
-------------

- **train**
  - Using a configuration, initialize a model and train it. Save the best model found as a model snapshot. This snapshot is something that can be loaded back in to PyText and trained further, tested, or exported.
- **test**
  - Load a trained model snapshot and evaluate its performance against a test set.
- **export**
  - Save the model as a serialized Caffe2 model, which is a stable model representation that can be loaded in production. (PyTorch model snapshots aren't very durable; if you update parts of your runtime environment, they may be invalidated).
- **predict**
  - Provide a simple REPL which lets you run inputs through your exported Caffe2 model and get a tangible sense for how your model will behave.

Train your first model
-------------------------

To get our feet wet, let's run one of the demo configurations included with PyText.

.. code-block:: console

  (pytext) $ cat demo/configs/docnn.json
  {
    "version": 8,
    "task": {
      "DocumentClassificationTask": {
        "data": {
          "source": {
            "TSVDataSource": {
              "field_names": ["label", "slots", "text"],
              "train_filename": "tests/data/train_data_tiny.tsv",
              "test_filename": "tests/data/test_data_tiny.tsv",
              "eval_filename": "tests/data/test_data_tiny.tsv"
            }
          }
        },
        "model": {
          "DocModel": {
            "representation": {
              "DocNNRepresentation": {}
            }
          }
        }
      }
    }
  }

This config will train a document classification model (DocNN) to detect the "class" of a series of commands given to a smart assistant. Let's take a quick look at the dataset:

.. code-block:: console

  (pytext) $ head -2 tests/data/train_data_tiny.tsv
  alarm/modify_alarm      16:24:datetime,39:57:datetime   change my alarm tomorrow to wake me up 30 minutes earlier
  alarm/set_alarm         Turn on all my alarms
  (pytext) $ wc -l tests/data/train_data_tiny.tsv
      10 tests/data/train_data_tiny.tsv

As you can see, the dataset is quite small, so don't get your hopes up on accuracy! We included this dataset for running unit tests against our models. PyText uses data in a tab separated format, as specified in the config by TSVDataSource. The order of the columns can be configured, but here we use the default. The first column is the "class", the output label that we're trying to predict. The second column is word-level tags, which we're not trying to predict yet, so ignore them for now. The last column here is the input text, which is the command whose class (the first column) the model tries to predict.

Let's train the model!

.. code-block:: console

  (pytext) $ pytext train < demo/configs/docnn.json
  ... [snip]

  Stage.TEST
  Epoch:1
  loss: 1.646484
  Accuracy: 50.00

  Soft Metrics:
  +--------------------------+-------------------+---------+
  | Label                    | Average precision | ROC AUC |
  +--------------------------+-------------------+---------+
  |       alarm/modify_alarm |               nan |   0.000 |
  |          alarm/set_alarm |             1.000 |   1.000 |
  |       alarm/snooze_alarm |               nan |   0.000 |
  | alarm/time_left_on_alarm |             0.333 |   0.333 |
  |    reminder/set_reminder |             1.000 |   1.000 |
  |  reminder/show_reminders |               nan |   0.000 |
  |             weather/find |               nan |   0.000 |
  +--------------------------+-------------------+---------+

  Recall at Precision
  +--------------------------+---------+---------+---------+---------+---------+
  | Label                    | R@P 0.2 | R@P 0.4 | R@P 0.6 | R@P 0.8 | R@P 0.9 |
  +--------------------------+---------+---------+---------+---------+---------+
  | alarm/modify_alarm       |   0.000 |   0.000 |   0.000 |   0.000 |   0.000 |
  | alarm/set_alarm          |   1.000 |   1.000 |   1.000 |   1.000 |   1.000 |
  | alarm/snooze_alarm       |   0.000 |   0.000 |   0.000 |   0.000 |   0.000 |
  | alarm/time_left_on_alarm |   1.000 |   0.000 |   0.000 |   0.000 |   0.000 |
  | reminder/set_reminder    |   1.000 |   1.000 |   1.000 |   1.000 |   1.000 |
  | reminder/show_reminders  |   0.000 |   0.000 |   0.000 |   0.000 |   0.000 |
  | weather/find             |   0.000 |   0.000 |   0.000 |   0.000 |   0.000 |
  +--------------------------+---------+---------+---------+---------+---------+
  saving result to file /tmp/test_out.txt

The model ran over the training set 10 times. This output is the result of evaluating the model on the test set, and tracking how well it did. If you're not familiar with these accuracy measurements,

- **Precision** - The number of times the model guessed this label and was right
- **Recall** - The number of times the model correctly identified this label, out of every time it shows up in the test set. If this number is low for a label, the model should be predicting this label more.
- **F1** - A harmonic mean of recall and precision.
- **Support** - The number of times this label shows up in the test set.

As you can see, the training results were pretty bad. We ran over the data 10 times, and in that time managed to learn how to predict only one of the labels in the test set successfully. In fact, many of the labels were never predicted at all! With 10 examples, that's not too surprising. See the next tutorial to run on a real dataset and get more usable results.
