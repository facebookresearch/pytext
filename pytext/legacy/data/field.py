#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# coding: utf8

from collections import Counter, OrderedDict
from itertools import chain

import torch
from torchtext.data.utils import get_tokenizer, dtype_to_attr, is_tokenizer_serializable

from ..vocab import Vocab
from .dataset import Dataset
from .pipeline import Pipeline


class RawField(object):
    """Defines a general datatype.

    Every dataset consists of one or more types of data. For instance, a text
    classification dataset contains sentences and their classes, while a
    machine translation dataset contains paired examples of text in two
    languages. Each of these types of data is represented by a RawField object.
    A RawField object does not assume any property of the data type and
    it holds parameters relating to how a datatype should be processed.

    Attributes:
        preprocessing: The Pipeline that will be applied to examples
            using this field before creating an example.
            Default: None.
        postprocessing: A Pipeline that will be applied to a list of examples
            using this field before assigning to a batch.
            Function signature: (batch(list)) -> object
            Default: None.
        is_target: Whether this field is a target variable.
            Affects iteration over batches. Default: False
    """

    def __init__(self, preprocessing=None, postprocessing=None, is_target=False):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.is_target = is_target

    def preprocess(self, x):
        """Preprocess an example if the `preprocessing` Pipeline is provided."""
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        """Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
            postprocessing Pipeline.
        """
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return batch


class Field(RawField):
    """Defines a datatype together with instructions for converting to Tensor.

    Field class models common text processing datatypes that can be represented
    by tensors.  It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.

    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.

    Attributes:
        sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        dtype: The torch.dtype class that represents a batch of examples
            of this kind of data. Default: torch.long.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: None.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list, and
            the field's Vocab.
            Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy tokenizer is
            used. If a non-serializable function is passed as an argument,
            the field will not be able to be serialized. Default: string.split.
        tokenizer_language: The language of the tokenizer to be constructed.
            Various languages currently supported only in SpaCy.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
        unk_token: The string token used to represent OOV words. Default: "<unk>".
        pad_first: Do the padding of the sequence at the beginning. Default: False.
        truncate_first: Do the truncating of the sequence at the beginning. Default: False
        stop_words: Tokens to discard during the preprocessing step. Default: None
        is_target: Whether this field is a target variable.
            Affects iteration over batches. Default: False
    """

    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,
        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }

    ignore = ["dtype", "tokenize"]

    def __init__(
        self,
        sequential=True,
        use_vocab=True,
        init_token=None,
        eos_token=None,
        fix_length=None,
        dtype=torch.long,
        preprocessing=None,
        postprocessing=None,
        lower=False,
        tokenize=None,
        tokenizer_language="en",
        include_lengths=False,
        batch_first=False,
        pad_token="<pad>",
        unk_token="<unk>",
        pad_first=False,
        truncate_first=False,
        stop_words=None,
        is_target=False,
    ):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        # store params to construct tokenizer for serialization
        # in case the tokenizer isn't picklable (e.g. spacy)
        self.tokenizer_args = (tokenize, tokenizer_language)
        self.tokenize = get_tokenizer(tokenize, tokenizer_language)
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token if self.sequential else None
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        try:
            self.stop_words = set(stop_words) if stop_words is not None else None
        except TypeError:
            raise ValueError("Stop words must be convertible to a set")
        self.is_target = is_target

    def __getstate__(self):
        str_type = dtype_to_attr(self.dtype)
        if is_tokenizer_serializable(*self.tokenizer_args):
            tokenize = self.tokenize
        else:
            # signal to restore in `__setstate__`
            tokenize = None
        attrs = {k: v for k, v in self.__dict__.items() if k not in self.ignore}
        attrs["dtype"] = str_type
        attrs["tokenize"] = tokenize

        return attrs

    def __setstate__(self, state):
        state["dtype"] = getattr(torch, state["dtype"])
        if not state["tokenize"]:
            state["tokenize"] = get_tokenizer(*state["tokenizer_args"])
        self.__dict__.update(state)

    def __hash__(self):
        # we don't expect this to be called often
        return 42

    def __eq__(self, other):
        if not isinstance(other, RawField):
            return False

        return self.__dict__ == other.__dict__

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.

        If `sequential=True`, the input will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if self.sequential and isinstance(x, str):
            x = self.tokenize(x.rstrip("\n"))
        if self.lower:
            x = Pipeline(str.lower)(x)
        if self.sequential and self.use_vocab and self.stop_words is not None:
            x = [w for w in x if w not in self.stop_words]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):
        """Process a list of examples to create a torch.Tensor.

        Pad, numericalize, and postprocess a batch and create a tensor.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
        """
        minibatch = list(minibatch)
        if not self.sequential:
            return minibatch
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = (
                self.fix_length + (self.init_token, self.eos_token).count(None) - 2
            )
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x))
                    + ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token])
                )
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token])
                    + [self.pad_token] * max(0, max_len - len(x))
                )
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return (padded, lengths)
        return padded

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for this field from one or more datasets.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for this field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [
                    getattr(arg, name)
                    for name, field in arg.fields.items()
                    if field is self
                ]
            else:
                sources.append(arg)
        for data in sources:
            for x in data:
                if not self.sequential:
                    x = [x]
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))
        specials = list(
            OrderedDict.fromkeys(
                tok
                for tok in [
                    self.unk_token,
                    self.pad_token,
                    self.init_token,
                    self.eos_token,
                ]
                + kwargs.pop("specials", [])
                if tok is not None
            )
        )
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])): List of tokenized
                and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError(
                "Field has include_lengths set to True, but "
                "input data is not a tuple of "
                "(data batch, batch lengths)."
            )
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype)
                )
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explicitly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [
                    numericalization_func(x) if isinstance(x, str) else x for x in arr
                ]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        var = torch.tensor(arr, dtype=self.dtype, device=device)

        if self.sequential and not self.batch_first:
            var.t_()
        if self.sequential:
            var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var


class NestedField(Field):
    """A nested field.

    A nested field holds another field (called *nesting field*), accepts an untokenized
    string or a list string tokens and groups and treats them as one field as described
    by the nesting field. Every token will be preprocessed, padded, etc. in the manner
    specified by the nesting field. Note that this means a nested field always has
    ``sequential=True``. The two fields' vocabularies will be shared. Their
    numericalization results will be stacked into a single tensor. And NestedField will
    share the same include_lengths with nesting_field, so one shouldn't specify the
    include_lengths in the nesting_field. This field is
    primarily used to implement character embeddings. See ``tests/data/test_field.py``
    for examples on how to use this field.

    Arguments:
        nesting_field (Field): A field contained in this nested field.
        use_vocab (bool): Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: ``True``.
        init_token (str): A token that will be prepended to every example using this
            field, or None for no initial token. Default: ``None``.
        eos_token (str): A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: ``None``.
        fix_length (int): A fixed length that all examples using this field will be
            padded to, or ``None`` for flexible sequence lengths. Default: ``None``.
        dtype: The torch.dtype class that represents a batch of examples
            of this kind of data. Default: ``torch.long``.
        preprocessing (Pipeline): The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: ``None``.
        postprocessing (Pipeline): A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list, and
            the field's Vocab. Default: ``None``.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy tokenizer is
            used. If a non-serializable function is passed as an argument,
            the field will not be able to be serialized. Default: string.split.
        tokenizer_language: The language of the tokenizer to be constructed.
            Various languages currently supported only in SpaCy.
        pad_token (str): The string token used as padding. If ``nesting_field`` is
            sequential, this will be set to its ``pad_token``. Default: ``"<pad>"``.
        pad_first (bool): Do the padding of the sequence at the beginning. Default:
            ``False``.
    """

    def __init__(
        self,
        nesting_field,
        use_vocab=True,
        init_token=None,
        eos_token=None,
        fix_length=None,
        dtype=torch.long,
        preprocessing=None,
        postprocessing=None,
        tokenize=None,
        tokenizer_language="en",
        include_lengths=False,
        pad_token="<pad>",
        pad_first=False,
        truncate_first=False,
    ):
        if isinstance(nesting_field, NestedField):
            raise ValueError("nesting field must not be another NestedField")
        if nesting_field.include_lengths:
            raise ValueError("nesting field cannot have include_lengths=True")

        if nesting_field.sequential:
            pad_token = nesting_field.pad_token
        super(NestedField, self).__init__(
            use_vocab=use_vocab,
            init_token=init_token,
            eos_token=eos_token,
            fix_length=fix_length,
            dtype=dtype,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            lower=nesting_field.lower,
            tokenize=tokenize,
            tokenizer_language=tokenizer_language,
            batch_first=True,
            pad_token=pad_token,
            unk_token=nesting_field.unk_token,
            pad_first=pad_first,
            truncate_first=truncate_first,
            include_lengths=include_lengths,
        )
        self.nesting_field = nesting_field
        # in case the user forget to do that
        self.nesting_field.batch_first = True

    def preprocess(self, xs):
        """Preprocess a single example.

        Firstly, tokenization and the supplied preprocessing pipeline is applied. Since
        this field is always sequential, the result is a list. Then, each element of
        the list is preprocessed using ``self.nesting_field.preprocess`` and the resulting
        list is returned.

        Arguments:
            xs (list or str): The input to preprocess.

        Returns:
            list: The preprocessed list.
        """
        return [
            self.nesting_field.preprocess(x)
            for x in super(NestedField, self).preprocess(xs)
        ]

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        If ``self.nesting_field.sequential`` is ``False``, each example in the batch must
        be a list of string tokens, and pads them as if by a ``Field`` with
        ``sequential=True``. Otherwise, each example must be a list of list of tokens.
        Using ``self.nesting_field``, pads the list of tokens to
        ``self.nesting_field.fix_length`` if provided, or otherwise to the length of the
        longest list of tokens in the batch. Next, using this field, pads the result by
        filling short examples with ``self.nesting_field.pad_token``.

        Example:
            >>> import pprint
            >>> pp = pprint.PrettyPrinter(indent=4)
            >>>
            >>> nesting_field = Field(pad_token='<c>', init_token='<w>', eos_token='</w>')
            >>> field = NestedField(nesting_field, init_token='<s>', eos_token='</s>')
            >>> minibatch = [
            ...     [list('john'), list('loves'), list('mary')],
            ...     [list('mary'), list('cries')],
            ... ]
            >>> padded = field.pad(minibatch)
            >>> pp.pprint(padded)
            [   [   ['<w>', '<s>', '</w>', '<c>', '<c>', '<c>', '<c>'],
                    ['<w>', 'j', 'o', 'h', 'n', '</w>', '<c>'],
                    ['<w>', 'l', 'o', 'v', 'e', 's', '</w>'],
                    ['<w>', 'm', 'a', 'r', 'y', '</w>', '<c>'],
                    ['<w>', '</s>', '</w>', '<c>', '<c>', '<c>', '<c>']],
                [   ['<w>', '<s>', '</w>', '<c>', '<c>', '<c>', '<c>'],
                    ['<w>', 'm', 'a', 'r', 'y', '</w>', '<c>'],
                    ['<w>', 'c', 'r', 'i', 'e', 's', '</w>'],
                    ['<w>', '</s>', '</w>', '<c>', '<c>', '<c>', '<c>'],
                    ['<c>', '<c>', '<c>', '<c>', '<c>', '<c>', '<c>']]]

        Arguments:
            minibatch (list): Each element is a list of string if
                ``self.nesting_field.sequential`` is ``False``, a list of list of string
                otherwise.

        Returns:
            list: The padded minibatch. or (padded, sentence_lens, word_lengths)
        """
        minibatch = list(minibatch)
        if not self.nesting_field.sequential:
            return super(NestedField, self).pad(minibatch)

        # Save values of attributes to be monkeypatched
        old_pad_token = self.pad_token
        old_init_token = self.init_token
        old_eos_token = self.eos_token
        old_fix_len = self.nesting_field.fix_length
        # Monkeypatch the attributes
        if self.nesting_field.fix_length is None:
            max_len = max(len(xs) for ex in minibatch for xs in ex)
            fix_len = (
                max_len
                + 2
                - (self.nesting_field.init_token, self.nesting_field.eos_token).count(
                    None
                )
            )
            self.nesting_field.fix_length = fix_len
        self.pad_token = [self.pad_token] * self.nesting_field.fix_length
        if self.init_token is not None:
            # self.init_token = self.nesting_field.pad([[self.init_token]])[0]
            self.init_token = [self.init_token]
        if self.eos_token is not None:
            # self.eos_token = self.nesting_field.pad([[self.eos_token]])[0]
            self.eos_token = [self.eos_token]
        # Do padding
        old_include_lengths = self.include_lengths
        self.include_lengths = True
        self.nesting_field.include_lengths = True
        padded, sentence_lengths = super(NestedField, self).pad(minibatch)
        padded_with_lengths = [self.nesting_field.pad(ex) for ex in padded]
        word_lengths = []
        final_padded = []
        max_sen_len = len(padded[0])
        for (pad, lens), sentence_len in zip(padded_with_lengths, sentence_lengths):
            if sentence_len == max_sen_len:
                lens = lens
                pad = pad
            elif self.pad_first:
                lens[: (max_sen_len - sentence_len)] = [0] * (
                    max_sen_len - sentence_len
                )
                pad[: (max_sen_len - sentence_len)] = [self.pad_token] * (
                    max_sen_len - sentence_len
                )
            else:
                lens[-(max_sen_len - sentence_len) :] = [0] * (
                    max_sen_len - sentence_len
                )
                pad[-(max_sen_len - sentence_len) :] = [self.pad_token] * (
                    max_sen_len - sentence_len
                )
            word_lengths.append(lens)
            final_padded.append(pad)
        padded = final_padded

        # Restore monkeypatched attributes
        self.nesting_field.fix_length = old_fix_len
        self.pad_token = old_pad_token
        self.init_token = old_init_token
        self.eos_token = old_eos_token
        self.include_lengths = old_include_lengths
        if self.include_lengths:
            return padded, sentence_lengths, word_lengths
        return padded

    def build_vocab(self, *args, **kwargs):
        """Construct the Vocab object for nesting field and combine it with this field's vocab.

        Arguments:
            Positional arguments: Dataset objects or other iterable data
                sources from which to construct the Vocab object that
                represents the set of possible values for the nesting field. If
                a Dataset object is provided, all columns corresponding
                to this field are used; individual columns can also be
                provided directly.
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources.extend(
                    [
                        getattr(arg, name)
                        for name, field in arg.fields.items()
                        if field is self
                    ]
                )
            else:
                sources.append(arg)

        flattened = []
        for source in sources:
            flattened.extend(source)
        old_vectors = None
        old_unk_init = None
        old_vectors_cache = None
        if "vectors" in kwargs.keys():
            old_vectors = kwargs["vectors"]
            kwargs["vectors"] = None
        if "unk_init" in kwargs.keys():
            old_unk_init = kwargs["unk_init"]
            kwargs["unk_init"] = None
        if "vectors_cache" in kwargs.keys():
            old_vectors_cache = kwargs["vectors_cache"]
            kwargs["vectors_cache"] = None
        # just build vocab and does not load vector
        self.nesting_field.build_vocab(*flattened, **kwargs)
        super(NestedField, self).build_vocab()
        self.vocab.extend(self.nesting_field.vocab)
        self.vocab.freqs = self.nesting_field.vocab.freqs.copy()
        if old_vectors is not None:
            self.vocab.load_vectors(
                old_vectors, unk_init=old_unk_init, cache=old_vectors_cache
            )

        self.nesting_field.vocab = self.vocab

    def numericalize(self, arrs, device=None):
        """Convert a padded minibatch into a variable tensor.

        Each item in the minibatch will be numericalized independently and the resulting
        tensors will be stacked at the first dimension.

        Arguments:
            arrs (List[List[str]]): List of tokenized and padded examples.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        numericalized = []
        self.nesting_field.include_lengths = False
        if self.include_lengths:
            arrs, sentence_lengths, word_lengths = arrs

        for arr in arrs:
            numericalized_ex = self.nesting_field.numericalize(arr, device=device)
            numericalized.append(numericalized_ex)
        padded_batch = torch.stack(numericalized)

        self.nesting_field.include_lengths = True
        if self.include_lengths:
            sentence_lengths = torch.tensor(
                sentence_lengths, dtype=self.dtype, device=device
            )
            word_lengths = torch.tensor(word_lengths, dtype=self.dtype, device=device)
            return (padded_batch, sentence_lengths, word_lengths)
        return padded_batch


class LabelField(Field):
    """A Label field.

    A label field is a shallow wrapper around a standard field designed to hold labels
    for a classification task. Its only use is to set the unk_token and sequential to
    `None` by default.
    """

    def __init__(self, **kwargs):
        # whichever value is set for sequential, unk_token, and is_target
        # will be overwritten
        kwargs["sequential"] = False
        kwargs["unk_token"] = None
        kwargs["is_target"] = True

        super(LabelField, self).__init__(**kwargs)
