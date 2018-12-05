Training your first model (part 1)
==================================

Once you've completed _`installation` or _`hack_on_pytext` you can start training your first model!

This tutorial series is an overview of *using* PyText, and will cover the main concepts and ways that PyText wants to interact with the world. It won't get into changing the code, for instance hacking on new model architectures. By the end, you should have a high-quality text model that can be used in production.

You can use PyText as a library either in your own scripts or in a notebook, but the fastest way to get started training is through the PyText command line tool. This tool will automatically be in your path when you install PyText!::

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

Right away you're presented with all of the most important parts of using PyText.

Background
==========
Fundamentally, machine learning is learning a function. Your training, evaluation, and test datasets are examples of inputs and their corresponding outputs showing how that function behaves. A **model** is an implementation of that function. To **train** a **model** means to make a statistical machine-learned implementation of that function that using the training data examples as a rubrik. To **predict** using a **model** means to take a trained implementation and apply it to new inputs, and predict what the result of the idealized function would be on those inputs.

More examples to train on generally corresponds to more accurate and better-generalizing models. This can mean thousands to millions or billions of examples depending on the task (function) you're trying to learn.

PyText Configs
==============
Creating a state-of-the-art model is mainly about configuration. Picking your training dataset, your model parameters, your training parameters, and so on, is a central part of building high-quality text models.

Configuration is a central part of every component within PyText, and the config system that we built allows for all of these configurations to be serialized and deserialized to JSON.

There are a number of example configurations that can train simple models provided with PyText, and we have a system for automatically documenting the default configurations and possible configuration values.

PyText Modes
============
- **train**
  - Using a configuration, initialize a model and train it. Save the best model found as a model snapshot. This snapshot is something that can be loaded back in to PyText and trained more, tested, or exported.
- **test**
  - Load a trained model snapshot and evaluate its performance against a test set.
- **export**
  - Model snapshots aren't very durable. If you update parts of your runtime environment, they may be invalidated. Export saves the model as a serialized pytorch caffe2 model, which is a stable model representation that will be able to be loaded in production.
- **predict**
  - This is an example of how your deployed code should interact with a model. In this case this is a simple REPL that will let you execute examples against your exported pytorch caffe2 model and get a tangible sense for how your model behaves.

Training your first model
=========================
To get our feet wet, let's run one of the demo configurations included with PyText.::

  (pytext) $ cat demo/configs/docnn.json
  {
    "task": {
      "DocClassificationTask": {
	"data_handler": {
	  "train_path": "tests/data/train_data_tiny.tsv",
	  "eval_path": "tests/data/test_data_tiny.tsv",
	  "test_path": "tests/data/test_data_tiny.tsv"
	}
      }
    }
  }

This trains a DocNN model (DocNN is a classification model architecture utilizing work embeddings and convolutional layers) to detect a series of commands based on a smart assistant. Let's take a quick look at the dataset: ::

  (pytext) $ head -2 tests/data/train_data_tiny.tsv
  alarm/modify_alarm      16:24:datetime,39:57:datetime   change my alarm tomorrow to wake me up 30 minutes earlier
  alarm/set_alarm         Turn on all my alarms
  (pytext) $ wc -l tests/data/train_data_tiny.tsv
      10 tests/data/train_data_tiny.tsv

As you can see, the dataset is quite small, so don't get your hopes up on accuracy! This is a dataset we've included for running unit tests against our models. PyText uses TSV format. The order of the columns can be configured, but this is the default. The first column is the "class", the output label that we're trying to predict. The second column is word-level tags, which we're not trying to predict now, so ignore them for now. The last column here is the input text, which is what the model is trying to use to predict the label (the first column).

Let's try training the model!
::

  (pytext) $ pytext train < demo/configs/docnn.json
  ... [snip]
    
  Stage.TEST
  loss: 2.072155
  Accuracy: 20.00

  Macro P/R/F1 Scores:
	  Label                   Precision       Recall          F1              Support

	  reminder/set_reminder   20.00           100.00          33.33           1
	  alarm/time_left_on_alarm        0.00            0.00            0.00            1
	  alarm/show_alarms       0.00            0.00            0.00            1
	  alarm/set_alarm         0.00            0.00            0.00            2
	  Overall macro scores    5.00            25.00           8.33

  Soft Metrics:
	  Label           Average precision
	  alarm/set_alarm 40.00
	  alarm/time_left_on_alarm        100.00
	  reminder/set_reminder   25.00
	  alarm/show_alarms       25.00
	  weather/find    nan
	  alarm/modify_alarm      nan
	  alarm/snooze_alarm      nan
	  reminder/show_reminders nan
  saving result to file /tmp/test_out.txt

The model ran over the training set 10 times. This output is the result of evaluating the model on the test set, and tracking how well it did. If you're not familiar with these accuracy measurements,

- **Precision** - The number of times the model guessed this label and was right
- **Recall** - How many times the model correctly identified this label out of every time it shows up in the test set. If this number is low for a label, the model should be guessing this label more.
- **F1** - A geometric average of recall and precision.
- **Support** - How many times this label shows up in the test set.

As you can see, the training results were pretty bad. We ran over the data 10 times, and in that time managed to learn how to predict one of the labels in the test set successfully. In fact, many of the labels were never output at all! With 10 examples, that's not too surprising. See the next tutorial to run on a real dataset and get some meaningful results.
