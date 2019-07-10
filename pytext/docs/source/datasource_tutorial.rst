Custom Data Format
==================

PyText's default reader is :class:`~TSVDataSource` to read your dataset if it's in tsv format (tab-separated values). In many cases, your data is going to be in a different format. You could write a pre-processing script to format your data into tsv format, but it's easier and more convenient to implement your own :class:`~DataSource` component so that PyText can read your data directly, without any preprocessing.

This tutorial explains how to implement a simple :class:`~DataSource` that can read the ATIS data and to perform a classification task using the "intent" labels.

1. Download the data
--------------------

Download the `ATIS (Airline Travel Information System) dataset <https://www.kaggle.com/siddhadev/ms-cntk-atis/downloads/atis.zip/3>`_ and unzip it in a directory. Note that to download the dataset, you will need a `Kaggle <https://www.kaggle.com/>`_ account for which you can sign up for free. The zip file is about 240KB.

.. code-block:: console

    $ unzip <download_dir>/atis.zip -d <download_dir>/atis

2. The data format
------------------

The ATIS dataset has a few defining characterics:

#. it has a train set and a test set, but not eval set
#. the data is split into a "dict" file, which is a vocab file containing the words or labels, and the train and test sets, which only contain integers representing the word indexes.
#. sentences always start with the token 178 = BOS (Beginning Of Sentence) and end with the token 179 = EOS (End Of Sentence).

.. code-block:: console

    $ tail atis/atis.dict.vocab.csv
    y
    year
    yes
    yn
    york
    you
    your
    yx
    yyz
    zone
    $ tail atis/atis.test.query.csv
    178 479 0 545 851 264 882 429 851 915 330 179
    178 479 902 851 264 180 428 444 736 521 301 851 915 330 179
    178 818 581 207 827 204 616 915 330 179
    178 479 0 545 851 264 180 428 444 299 851 619 937 301 654 887 200 435 621 740 179
    178 818 581 207 827 204 482 827 619 937 301 229 179
    178 688 423 207 827 429 444 299 851 218 203 482 827 619 937 301 229 826 236 621 740 253 130 689 179
    178 423 581 180 428 444 299 851 218 203 482 827 619 937 301 229 179
    178 479 0 545 851 431 444 589 851 297 654 212 200 179
    178 479 932 545 851 264 180 730 870 428 444 511 301 851 297 179
    178 423 581 180 428 826 427 444 587 851 810 179

Our :class:`~DataSource` must then resolve the words from the vocab files to rebuild the sentences and labels as strings. It must also take a subset the one of train or test dataset to create the eval dataset. Since the test set is pretty small, we'll use the train set for that purpose and randomly take a small fraction (say 25%) to create the eval set. Finally, we can safely remove the first and last tokens of every query (BOS and EOS), as they don't add any value for classification.

The ATIS dataset also has information for slots tagging that we'll ignore because we only care about classification in this tutorial.

3. DataSource
-------------

PyText defines a :class:`~DataSource` to read the data. It expect each row of data to be represented as a python dict where the keys are the column names and the values are the columns properly typed.

Most of the time, the dataset will come as strings and the casting to the proper types can be inferred automatically from the other components in the config. To make the implementation of a new :class:`~DataSource` easier, PyText provides the class :class:`~RootDataSource` that does this type lookup for you. Most users should use :class:`~RootDataSource` as a base class.

4. Implementing `AtisIntentDataSource`
--------------------------------------

We will write the all the code for our :class:`~AtisIntentDataSource` in the file `my_classifier/source.py`.

First, let's write the utilities that will help us read the data: a function to load the vocab files, and the generator that uses them to rebuild the sentences and labels. We return pytext.data.utils.UNK for unknown words. We store the indexes as strings to avoid casting from and to ints when reading the inputs.::

  def load_vocab(file_path):
      """
      Given a file, prepare the vocab dictionary where each line is the value and
      (line_no - 1) is the key
      """
      vocab = {}
      with open(file_path, "r") as file_contents:
          for idx, word in enumerate(file_contents):
              vocab[str(idx)] = word.strip()
      return vocab

  def reader(file_path, vocab):
      with open(file_path, "r") as reader:
          for line in reader:
              yield " ".join(
                  vocab.get(s.strip(), UNK)
                  # ATIS every row starts/ends with BOS/EOS: remove them
                  for s in line.split()[1:-1]
              )


Then we declate the :class:`~DataSource` class itself: :class:`~AtisIntentDataSource`. It inherits from :class:`~RootDataSource`, which gives us the automatic lookup of data types. We declare all the config parameters that will be useful, and give sensible default values so that the general case where users provide only `path` and `field_names` will likely work. We load the vocab files for queries and intent only once in the constructor and keep them in memory for the entire run::

    class AtisIntentDataSource(RootDataSource):

        def __init__(
            self,
            path="my_directory",
            field_names=None,
            validation_split=0.25,
            random_seed=12345,
            # Filenames can be overridden if necessary
            intent_filename="atis.dict.intent.csv",
            vocab_filename="atis.dict.vocab.csv",
            test_queries_filename="atis.test.query.csv",
            test_intent_filename="atis.test.intent.csv",
            train_queries_filename="atis.train.query.csv",
            train_intent_filename="atis.train.intent.csv",
            **kwargs,
        ):
            super().__init__(**kwargs)

            field_names = field_names or ["text", "label"]
            assert len(field_names or []) == 2, \
               "AtisIntentDataSource only handles 2 field_names: {}".format(field_names)

            self.random_seed = random_seed
            self.eval_split = eval_split

            # Load the vocab dict in memory for the readers
            self.words = load_vocab(os.path.join(path, vocab_filename))
            self.intents = load_vocab(os.path.join(path, intent_filename))

            self.query_field = field_names[0]
            self.intent_field = field_names[1]

            self.test_queries_filepath = os.path.join(path, test_queries_filename)
            self.test_intent_filepath = os.path.join(path, test_intent_filename)
            self.train_queries_filepath = os.path.join(path, train_queries_filename)
            self.train_intent_filepath = os.path.join(path, train_intent_filename)

To generate the eval data set, we need to randomly select some of the rows in training, but in a consistent and repeatable way. This is not strictly needed, and the training will work if the selection were completely random, but having a consistent sequence will help with debugging and give comparable results from training to training. In order to do that, we need to use the same seed for a new random number generator each time we start reading the train data set. The function below can be used for either training or eval and ensures that those two sets are complement of each other, with the ratio determined by eval_split. This function returns True or False depending on whether the row should be included or not::

        def _selector(self, select_eval):
            """
            This selector ensures that the same pseudo-random sequence is
            always the used from the Beginning. The `select_eval` parameter
            guarantees that the training set and eval set are exact complements.
            """
            rng = Random(self.random_seed)
            def fn():
                return select_eval ^ (rng.random() >= self.eval_split)
            return fn

Next, we write the function that iterates through both the `reader` for the queries (sentences) and the `reader` for the intents (labels) simultaneously. It yields each row in the form a python dictionnary, where the keys are the `field_names`. We can pass an optional function to select a subset of the row (ie: _selector defined above); the default is to select all the rows::

        def _iter_rows(self, query_reader, intent_reader, select_fn=lambda: True):
            for query_str, intent_str in zip(query_reader, intent_reader):
                if select_fn():
                    yield {
                        # in ATIS every row starts/ends with BOS/EOS: remove them
                        self.query_field: query_str[4:-4],
                        self.intent_field: intent_str,
                    }

Finally, we tie everything toghether by implementing the 3 API methods of :class:`~RootDataSource`. Each of those methods should return a generator that can iterate through the specific dataset entirely. For the test dataset, we simply return all the row presented by the data in test_queries_filepath and test_intent_filepath, using the corresponding vocab::

        def raw_test_data_generator(self):
            return iter(self._iter_rows(
                query_reader=reader(
                    self.test_queries_filepath,
                    self.words,
                ),
                intent_reader=reader(
                    self.test_intent_filepath,
                    self.intents,
                ),
            ))

For the eval and train datasets, we read the same files train_queries_filepath and train_intent_filepath, but we select some of the rows for eval and the rest for train::

        def raw_train_data_generator(self):
            return iter(self._iter_rows(
                query_reader=reader(
                    self.train_queries_filepath,
                    self.words,
                ),
                intent_reader=reader(
                    self.train_intent_filepath,
                    self.intents,
                ),
                select_fn=self._selector(select_eval=False),
            ))

        def raw_eval_data_generator(self):
            return iter(self._iter_rows(
                query_reader=reader(
                    self.train_queries_filepath,
                    self.words,
                ),
                intent_reader=reader(
                    self.train_intent_filepath,
                    self.intents,
                ),
                select_fn=self._selector(select_eval=True),
            ))

:class:`~RootDataSource` needs to know how it should transform the values in the dictionnaries created by the raw generators into the types matching the tensorizers used in the model. Fortunately, :class:`~RootDataSource` already provides a number of type conversion functions like the one below, so we don't need to do it for strings. If we did need to do it, we would declare one like this for :class:`~AtisIntentDataSource`.::

    @AtisIntentDataSource.register_type(str)
    def load_string(s):
        return s

The full source code for this tutorial can be found in `demo/datasource/source.py`, which include the `imports` needed.

5. Testing `AtisIntentDataSource`
---------------------------------

For rapid dev-test cycles, we add a simple main code printing the generated data in the terminal::

    if __name__ == "__main__":
        import sys
        src = AtisIntentDataSource(
            path=sys.argv[1],
            field_names=["query", "intent"],
            schema={},
        )
        for row in src.raw_train_data_generator():
            print("TRAIN", row)
        for row in src.raw_eval_data_generator():
            print("EVAL", row)
        for row in src.raw_test_data_generator():
            print("TEST", row)

We test our class to make sure we're getting the right data.

.. code-block:: console

    $ python3 my_classifier/source.py atis | head -n 3
    TRAIN {'query': 'what flights are available from pittsburgh to baltimore on thursday morning', 'intent': 'flight'}
    TRAIN {'query': 'cheapest airfare from tacoma to orlando', 'intent': 'airfare'}
    TRAIN {'query': 'round trip fares from pittsburgh to philadelphia under 1000 dollars', 'intent': 'airfare'}

    $ python3 my_classifier/source.py atis | cut -d " " -f 1 | uniq -c
    3732 TRAIN
    1261 EVAL
     893 TEST

6. Training the Model
---------------------

First let's get a config using our new :class:`~AtisIntentDataSource`

.. code-block:: console

    $ pytext --include my_classifier gen-default-config DocumentClassificationTask AtisIntentDataSource > my_classifier/config.json
    Including: my_classifier
    ... importing module: my_classifier.source
    ... importing: <class 'my_classifier.source.AtisIntentDataSource'>
    INFO - Applying option: task->data->source = AtisIntentDataSource

This default config contains all the parameters with their default value. So we edit the config to remove the parameters that we don't care about, and we edit the ones we care about. We only want to run 3 epochs for now. It looks like this.

.. code-block:: console

    $ cat my_classifier/config.json
    {
      "debug_path": "my_classifier.debug",
      "export_caffe2_path": "my_classifier.caffe2.predictor",
      "export_onnx_path": "my_classifier.onnx",
      "save_snapshot_path": "my_classifier.pt",
      "task": {
        "DocumentClassificationTask": {
          "data": {
            "Data": {
              "source": {
                "AtisIntentDataSource": {
                  "field_names": ["text", "label"],
                  "path": "atis",
                  "random_seed": 12345,
                  "validation_split": 0.25
                }
              }
            }
          },
          "metric_reporter": {
            "output_path": "my_classifier.out"
          },
          "trainer": {
            "epochs": 3
          }
        }
      },
      "test_out_path": "my_classifier_test.out",
      "version": 12
    }

And, at last, we can train the model

.. code-block:: console

    $ pytext --include my_classifier train < my_classifier/config.json

Notes
-----

In the current version of PyText, we need to explicitly declare a few more things, like the `Config` class (that looks like the __init__ parameters) and the from_config method::

        class Config(RootDataSource.Config):
            path: str = "."
            field_names: List[str] = ["text", "label"]
            validation_split: float = 0.25
            random_seed: int = 12345
            # Filenames can be overridden if necessary
            intent_filename: str = "atis.dict.intent.csv"
            vocab_filename: str = "atis.dict.vocab.csv"
            test_queries_filename: str = "atis.test.query.csv"
            test_intent_filename: str = "atis.test.intent.csv"
            train_queries_filename: str = "atis.train.query.csv"
            train_intent_filename: str = "atis.train.intent.csv"

        # Config mimics the constructor
        # This will be the default in future pytext.
        @classmethod
        def from_config(cls, config: Config, schema: Dict[str, Type]):
            return cls(schema=schema, **config._asdict())

