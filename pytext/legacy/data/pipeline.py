#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


class Pipeline(object):
    """Defines a pipeline for transforming sequence data.

    The input is assumed to be utf-8 encoded `str`.

    Attributes:
        convert_token: The function to apply to input sequence data.
        pipes: The Pipelines that will be applied to input sequence
            data in order.
    """

    def __init__(self, convert_token=None):
        """Create a pipeline.

        Arguments:
            convert_token: The function to apply to input sequence data.
                If None, the identity function is used. Default: None
        """
        if convert_token is None:
            self.convert_token = Pipeline.identity
        elif callable(convert_token):
            self.convert_token = convert_token
        else:
            raise ValueError(
                "Pipeline input convert_token {} is not None "
                "or callable".format(convert_token)
            )
        self.pipes = [self]

    def __call__(self, x, *args):
        """Apply the the current Pipeline(s) to an input.

        Arguments:
            x: The input to process with the Pipeline(s).
            Positional arguments: Forwarded to the `call` function
                of the Pipeline(s).
        """
        for pipe in self.pipes:
            x = pipe.call(x, *args)
        return x

    def call(self, x, *args):
        """Apply _only_ the convert_token function of the current pipeline
        to the input. If the input is a list, a list with the results of
        applying the `convert_token` function to all input elements is
        returned.

        Arguments:
            x: The input to apply the convert_token function to.
            Positional arguments: Forwarded to the `convert_token` function
                of the current Pipeline.
        """
        if isinstance(x, list):
            return [self.convert_token(tok, *args) for tok in x]
        return self.convert_token(x, *args)

    def add_before(self, pipeline):
        """Add a Pipeline to be applied before this processing pipeline.

        Arguments:
            pipeline: The Pipeline or callable to apply before this
                Pipeline.
        """
        if not isinstance(pipeline, Pipeline):
            pipeline = Pipeline(pipeline)
        self.pipes = pipeline.pipes[:] + self.pipes[:]
        return self

    def add_after(self, pipeline):
        """Add a Pipeline to be applied after this processing pipeline.

        Arguments:
            pipeline: The Pipeline or callable to apply after this
                Pipeline.
        """
        if not isinstance(pipeline, Pipeline):
            pipeline = Pipeline(pipeline)
        self.pipes = self.pipes[:] + pipeline.pipes[:]
        return self

    @staticmethod
    def identity(x):
        """Return a copy of the input.

        This is here for serialization compatibility with pickle.
        """
        return x
