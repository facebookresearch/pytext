Custom Tensorizer
=================

:class:`~Tensorizer` is the class that prepares your data coming out of the data source and transforms it into tensors suitable for processing. Each tensorizer knows how to prepare the input data from specific columns. In order to do that, the tensorizer (after initialization, such as creating or loading the vocabulary for look-ups) executes the following steps:

#. Its ``Config`` defines which column name(s) the tensorizer will look at
#. ``numberize()`` takes one row and transform the strings into numbers
#. ``tensorize()`` takes a batch of rows and creates the tensors

PyText provides a number of tensorizers for the most common cases. However, if you have your own custom features that don't have a suitable :class:`~Tensorizer`, you will need to write your own class. Fortunately it's quite easy: you simply need to create a class that inherits from :class:`~Tensorizer` (or one of its subclasses), and implement a few functions.

First a :class:`~Config` inner class, from_config class method, and the constructor `__init__`. This is just to declare member variables.

The tensorizer should declare the schema of your :class:`~Tensorizer` by defining a `column_schema` property which returns a list of tuples, one for each field/column read from the data source. Each tuple specifies the name of the column, and the type of the data. By specifying the type of your data, the data source will automatically parse the inputs and pass objects of those types to the tensorizers. You don't need to parse your own inputs.

For example, :class:`~SeqTokenTensorizer` reads one column from the input data. The data is formatted like a json list of strings: `["where do you wanna meet?", "MPK"]`. The schema declaration is like this:

.. code:: python

    @property
    def column_schema(self):
        return [(self.column, List[str])]


Another example with :class:`~GazetteerTensorizer`: it needs 2 columns, one string for the text itself, and one for the gazetteer features formatted like a complex json object. (The Gazetteer type is registered in the data source to automatically convert the raw strings from the input to this type.) The schema declaration is like this:

.. code:: python

    Gazetteer = List[Dict[str, Dict[str, float]]]

    @property
    def column_schema(self):
        return [(self.text_column, str), (self.dict_column, Gazetteer)]


Example Implementation
----------------------

Let's implement a simple word tensorizer that creates a tensor with the word indexes from a vocabulary.

.. code:: python

    class MyWordTensorizer(Tensorizer):

        class Config(Tensorizer.Config):
            #: The name of the text column to read from the data source.
            column: str = "text"

        @classmethod
        def from_config(cls, config: Config):
            return cls(column=config.column)

        def __init__(self, column):
            self.column = column
            self.vocab = vocab

        @property
        def column_schema(self):
            return [(self.column, str)]

Next we need to build the vocabulary by reading the training data and count the words. Since multiple tensorizers might need to read the data, we parallelize the reading part and the tensorizers use the pattern `row = yield` to read their inputs. In this simple example, our "tokenize" function is just going to split on spaces.

.. code:: python

    def _tokenize(self, row):
        raw_text = row[self.column]
        return raw_text.split()

    def initialize(self):
        """Build vocabulary based on training corpus."""
        vocab_builder = VocabBuilder()

        try:
            while True:
                row = yield
                words = _tokenize(row)
                vocab_builder.add_all(words)
        except GeneratorExit:
            self.vocab = vocab_builder.make_vocab()

The most important method is numberize, which takes a row and transforms it into list of numbers. The exact meaning of those numbers is arbitrary and depends on the design of the model. In our case, we look up the word indexes in the vocabulary.

.. code:: python

    def numberize(self, row):
        """Look up tokens in vocabulary to get their corresponding index"""
        words = _tokenize(row)
        idx = self.vocab.lookup_all(words)
        # LSTM representations need the length of the sequence
        return idx, len(idx)

Because LSTM-based representations need the length of the sequence to only consider the useful values and ignore the padding, we also return the length of each sequence.

Finally, the last function will create properly padded torch.Tensors from the batches produced by `numberize`. Numberized results can be cached for performance. We have a separate function to tensorize them because they are shuffled and batched differently (at each epoch), and then they will need different padding (because padding dimensions depend on the batch).

.. code:: python

    def tensorize(self, batch):
        tokens, seq_lens = zip(*batch)
        return (
            pad_and_tensorize(tokens, self.vocab.get_pad_index()),
            pad_and_tensorize(seq_lens),
        )

LSTM-based representations implemented in Torch also need the batches to be sorted by sequence length descending, so we're add in a sort function.

.. code:: python

    def sort_key(self, row):
        # LSTM representations need the batches to be sorted by descending seq_len
        return row[1]

The full code is in `demo/examples/tensorizer.py`


Testing
-------

We can test our tensorizer with the following code that initializes the vocab, then tries the `numberize` function:


.. code:: python

    rows = [
        {"text": "I want some coffee"},
        {"text": "Turn it up"},
    ]
    tensorizer = MyWordTensorizer(column="text")

    # Vocabulary starts with 0 and 1 for Unknown and Padding.
    # The rest of the vocabulary is built by the rows in order.
    init = tensorizer.initialize()
    init.send(None)  # start the loop
    for row in rows:
        init.send(row)
    init.close()

    # Verify numberize.
    numberized_rows = (tensorizer.numberize(r) for r in rows)
    words, seq_len = next(numberized_rows)
    assert words == [2, 3, 4, 5]
    assert seq_len == 4  # "I want some coffee" has 4 words
    words, seq_len = next(numberized_rows)
    assert words == [6, 7, 8]
    assert seq_len == 3  # "Turn it up" has 3 words

    # test again, this time also make the tensors
    numberized_rows = (tensorizer.numberize(r) for r in rows)
    words_tensors, seq_len_tensors = tensorizer.tensorize(numberized_rows)
    # Notice the padding (1) of the 2nd tensor to match the dimension
    assert words_tensors.equal(torch.tensor([[2, 3, 4, 5], [6, 7, 8, 1]]))
    assert seq_len_tensors.equal(torch.tensor([4, 3]))
