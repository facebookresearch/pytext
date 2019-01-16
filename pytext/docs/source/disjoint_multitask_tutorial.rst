Multitask training with disjoint datasets
===============================================

In this tutorial, we will jointly train a classification task with a language modeling task in a multitask setting. The models will share the embedding and representation layers.

We will use the following datasets:

1. Binarized Stanford Sentiment Treebank (SST-2), which is part of the `GLUE benchmark <https://gluebenchmark.com/>`_.  This dataset contains segments from movie reviews labeled with their binary sentiment.
2. `WikiText-2 <https://einstein.ai/research/blog/the-wikitext-long-term-dependency-language-modeling-dataset>`_, a medium-size language modeling dataset with text extracted from Wikipedia.


1. Fetch and prepare the dataset
----------------------------------

Download the dataset in a local directory. We will refer to this as `base_dir` in the next section.

.. code-block:: console

	$ curl "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip" -o wikitext-2-v1.zip
	$ unzip wikitext-2-v1.zip
	$ curl "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8" -o SST-2.zip
	$ unzip SST-2.zip

Remove headers from SST-2 data:

.. code-block:: console

  $ cd base_dir/SST-2
  $ sed -i '1d' train.tsv
  $ sed -i '1d' dev.tsv

Remove empty lines from WikiText:

.. code-block:: console

  $ cd base_dir/wikitext-2
  $ sed -i '/^\s*$/d' train.tsv
  $ sed -i '/^\s*$/d' valid.tsv
  $ sed -i '/^\s*$/d' test.tsv


2. Train a base model
-----------------------------

Prepare the configuration file for training. A sample config file for the base document classification model can be found in your PyText repository at ``demo/configs/sst2.json``. If you haven't set up PyText, please follow :doc:`installation`, then make the following changes in the config:

- Set `train_path` to `base_dir/SST-2/train.tsv`.
- Set `eval_path` to `base_dir/SST-2/eval.tsv`.
- Set `test_path` to `base_dir/SST-2/test.tsv`.

The test set labels for this tasks are not openly available, therefore we will use the dev set.
Train the model using the command below.

.. code-block:: console

	(pytext) $ pytext train < demo/configs/sst2.json

The output will look like:
::

  Stage.EVAL
  loss: 0.472868
  Accuracy: 85.67


3. Configure for multitasking
-----------------------------

The example configuration for this tutorial is at ``demo/configs/multitask_sst_lm.json``.
The main configuration is under `tasks`, which is a dictionary of task name to task config:
::

	"task_weights": {
		"SST2": 1,
		"LM": 1
	},
  "tasks": {
    "SST2": {
      "DocClassificationTask": { ... }
    },
    "LM": {
      "LMTask": { ... }
    }
  }

You can also modify `task_weights` to weight the loss for each task.
The sub-tasks can be configured as you would in a single task setting, with the exception of changes described in the next sections.


3. Specify which parameters to share
--------------------------------------

Parameter sharing is specified at module level with the `shared_module_key` parameter, which is an arbitrary string. Modules with identical `shared_module_key` share parameters.

Here we will share the BiLSTM module.  Under the `SST` task, we set
::

  "representation": {
    "BiLSTMDocAttention": {
      "lstm": {
        "shared_module_key": "SHARED_LSTM"
      }
    }
  }

Under the `LM` task, we set
::

  "representation": {
    "shared_module_key": "SHARED_LSTM"
  },

In this case, `BiLSTMDocAttention.lstm` of :class:`~DocClassificationTask` and `representation` of :class:`~LMTask` are both of type `BiLSTM`, therefore parameter sharing is possible.


3. Share the embedding layer
---------------------------------

The embedding is also a module, and can be similarly shared. This is configured under the `features` section. However, we need to ensure that we use the same vocabulary for both tasks, by specifying a pre-built vocabulary file. First create the vocabulary from the classification task data:

.. code-block:: console

  $ cd base_dir/SST-2
  $ cat train.tsv dev.tsv | tr ' ' '\n' | sort | uniq > sst_vocab.txt

Then point to this file in configuration:
::

  "features": {
      "shared_module_key": "SHARED_EMBEDDING",
      "word_feat": {
        "vocab_file": "base_dir/SST-2/sst_vocab.txt",
        "vocab_size": 15000,
        "vocab_from_train_data": false
      }
    }


3. Train the model
--------------------

You can train the model with

.. code-block:: console

	(pytext) $ pytext train < demo/configs/multitask_sst_lm.json

The output will look like
::

  Stage.EVAL
  loss: 0.455871
  Accuracy: 86.12

Not a great improvement, but we used a very primitive language modeling task (bi-directional with no masking) for the purposes of this tutorial. Happy multitasking!
