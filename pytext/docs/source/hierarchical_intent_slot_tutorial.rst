Hierarchical intent and slot filling.
===============================================

In this tutorial, we will train a semantic parser for task oriented dialog by modeling hierarchical intent and slots (`Gupta et al.`_, Semantic Parsing for Task Oriented Dialog using Hierarchical Representations, EMNLP 2018). The model we will use is Recurrent Neural Network Grammar (`Dyer et al.`_, Recurrent Neural Network Grammar, NAACL 2016) or RNNG for this. RNNG is a neural constituency parsing algorithm that explicitly models compositional structure of a sentence. It is able to learn about hierarchical relationship among the words and phrases in a given sentence thereby learning the underlying tree structure. The paper proposes generative as well as discriminative approaches. In this tutorial, we have implemented the discriminative approach.

This tutorial covers:

1. Train an RNNG model with TOP dataset (`Gupta et al.`_).
2. Load the trained model and run inference on the trained model.


1. Fetch the dataset
--------------------

Download the dataset in the directory you want to save it in. We will refer to this as ``base_dir`` in the next section.
::
	$ curl -o top-dataset-semantic-parsing.zip -L https://fb.me/semanticparsingdialog
	$ unzip top-dataset-semantic-parsing.zip


2. Prepare configuration file
-----------------------------

Prepare the configuration file for training. A sample config file can be found in your PyText repository at ``<pytext_root_directory>/demo/configs/rnng.json``. If you haven't set up PyText, please follow :doc:`installation`.

For this tutorial, please change the following in ``<pytext_root_dir>/demo/configs/rnng.json``.

- Set ``train_path`` to ``base_dir/top-dataset-semantic-parsing/train.tsv``
- Set ``eval_path`` to ``base_dir/top-dataset-semantic-parsing/eval.tsv``.
- Set ``test_path`` to ``base_dir/top-dataset-semantic-parsing/test.tsv``.


3. Train model with the downloaded dataset
------------------------------------------

Train the model using the command below
::
	$ cd <pytext_root_directory>
	$ pytext train < demo/configs/rnng.json


The output will look like:
::
	Merged Intent and Slot Metrics
	P = 24.03 R = 31.90, F1 = 27.41.

This will take about hour. If you want to train with a smaller dataset to make it quick then get a subset of dataset using the commands below and update the ``train_path``, ``eval_path`` and ``test_path`` in ``<pytext_root_dir>/demo/configs/rnng.json`` with the new files created with subset of data.
::
	head -n 1000 base_dir/top-dataset-semantic-parsing/train.tsv > base_dir/top-dataset-semantic-parsing/train_small.tsv
	head -n 100 base_dir/top-dataset-semantic-parsing/eval.tsv > base_dir/top-dataset-semantic-parsing/eval_small.tsv
	head -n 100 base_dir/top-dataset-semantic-parsing/test.tsv > base_dir/top-dataset-semantic-parsing/test_small.tsv

If you now train the model with smaller datasets, the output will look like:
::
	Merged Intent and Slot Metrics
	P = 24.03 R = 31.90, F1 = 27.41.

4. Test the model interactively against input utterances.
---------------------------------------------------------

Load the model using the command below
::
	$ pytext predict-py --model-file=/tmp/model.pt
	"please input a json example, the names should be the same with column_to_read in model training config:

This will give you REPL prompt. You can enter an utterance to get back the model's prediction repeatedly. You should enter in a json format shown below. Once done press Ctrl+D.
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

You can keep entering utterances to see the output.

.. _`Dyer et al.`: https://arxiv.org/abs/1602.07776
.. _`Gupta et al.`: https://arxiv.org/abs/1810.07942d
