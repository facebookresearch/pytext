Installation
============

PyText is available in the Python Package Index via

.. code-block:: console

  $ pip install pytext

**PyText currently only supports Python 3.6!** Make sure that you're using a Python 3.6 version of `pip` to install. The easiest way to get started on most systems is to create a `virtualenv`

.. code-block:: console

  $ python.3.6 -m virtualenv pytext
  $ source pytext/bin/activate
  (pytext) $ pip install pytext

This will install a version of PyTorch depending on your system. See `PyTorch <https://pytorch.org>`_ for more information. If you are using MacOS or Windows, this likely will not include GPU support by default; if you are using Linux, you should automatically get a version of PyTorch compatible with CUDA 9.0.

If you need a different version of PyTorch, install PyText, and then follow the instructions on the `PyTorch website <https://pytorch.org>`_ For instance, to install GPU support on Windows

.. code-block:: console

  (pytext) $ pip install pytext
  # This will overwrite your torch installation with a torch that uses CUDA 9.0
  (pytext) $ pip uninstall torch_nightly
  (pytext) $ pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html
