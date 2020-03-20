Semantic parsing with sequence-to-sequence models
=================================================

Introduction
------------

PyText provides an encoder-decoder framework that is suitable for any task
that requires mapping a sequence of input tokens to a sequence of output
tokens. The default implementation is based on recurrent neural networks
(RNNs), which have been shown to be `unreasonably effective
<http://karpathy.github.io/2015/05/21/rnn-effectiveness/>`_ at sequence
processing tasks. The default implementation includes three major components

#. A bidirectional LSTM sequence encoder 
#. An LSTM sequence decoder 
#. A sequence generator that supports incremental decoding and beam search

All of these components are Torchscript-friendly, so that the trained model
can be exported directly as-is.  Following the general design of PyText, each
of these components may be customized via their respective config objects or
replaced entirely by custom components.

Tutorial
--------

`Tutorial in notebook <https://github.com/facebookresearch/pytext/blob/master/demo/notebooks/seq2seq_tutorial.ipynb>`_
`Run the tutorial in Google Colab <https://colab.research.google.com/github/facebookresearch/pytext/blob/master/demo/notebooks/seq2seq_tutorial.ipynb>`_
