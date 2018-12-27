Hierarchical intent and slot filling
===============================================

In this tutorial, we will train a semantic parser for task oriented dialog by modeling hierarchical intents and slots (`Gupta et al. , Semantic Parsing for Task Oriented Dialog using Hierarchical Representations, EMNLP 2018 <https://arxiv.org/abs/1810.07942>`_). The underlying model used in the paper is the Recurrent Neural Network Grammar (`Dyer et al., Recurrent Neural Network Grammar, NAACL 2016 <https://arxiv.org/abs/1602.07776>`_). RNNG is neural constituency parser that explicitly models the compositional tree structure of the words and phrases in an utterance.

1. Fetch the dataset
--------------------

Download the dataset to a local directory. We will refer to this as `base_dir` in the next section.

.. code-block:: console

	$ curl -o top-dataset-semantic-parsing.zip -L https://fb.me/semanticparsingdialog
	$ unzip top-dataset-semantic-parsing.zip


2. Prepare configuration file
-----------------------------

Prepare the configuration file for training. A sample config file can be found in your PyText repository at ``demo/configs/rnng.json``. If you haven't set up PyText, please follow :doc:`installation`, then make the following changes in the config:

- Set `train_path` to `base_dir/top-dataset-semantic-parsing/train.tsv`.
- Set `eval_path` to `base_dir/top-dataset-semantic-parsing/eval.tsv`.
- Set `test_path` to `base_dir/top-dataset-semantic-parsing/test.tsv`.


3. Train a model with the downloaded dataset
--------------------------------------------

Train the model using the command below

.. code-block:: console

	(pytext) $ pytext train < demo/configs/rnng.json


The output will look like:

.. code-block:: console

	Merged Intent and Slot Metrics
	P = 24.03 R = 31.90, F1 = 27.41.

This will take about hour. If you want to train with a smaller dataset to make it quick then generate a subset of the dataset using the commands below and update the paths in ``demo/configs/rnng.json``:

.. code-block:: console

	$ head -n 1000 base_dir/top-dataset-semantic-parsing/train.tsv > base_dir/top-dataset-semantic-parsing/train_small.tsv
	$ head -n 100 base_dir/top-dataset-semantic-parsing/eval.tsv > base_dir/top-dataset-semantic-parsing/eval_small.tsv
	$ head -n 100 base_dir/top-dataset-semantic-parsing/test.tsv > base_dir/top-dataset-semantic-parsing/test_small.tsv

If you now train the model with smaller datasets, the output will look like:

.. code-block:: console

	Merged Intent and Slot Metrics
	P = 24.03 R = 31.90, F1 = 27.41.

4. Test the model interactively against input utterances.
---------------------------------------------------------

Load the model using the command below

.. code-block:: console

	(pytext) $ pytext predict-py --model-file=/tmp/model.pt
	please input a json example, the names should be the same with column_to_read in model training config:

This will give you a REPL prompt. You can enter an utterance to get back the model's prediction repeatedly. You should enter in a json format shown below. Once done press Ctrl+D.
::

	{"text": "order coffee from starbucks"}

You should see an output like:
::

	[{'prediction': [7, 0, 5, 0, 1, 0, 3, 0, 1, 1],
	'score': [
		0.44425372408062447,
		0.8018286800064633,
		0.6880680051949267,
		0.9891564979506277,
		0.9999506231665385,
		0.9992705616574005,
		0.34512090135492923,
		0.9999979545618913,
		0.9999998668826438,
		0.9999998686418744]}]

We have also provided a pre-trained model which you may download `here <https://download.pytorch.org/data/rnng_topv1.1_release.pt>`_
