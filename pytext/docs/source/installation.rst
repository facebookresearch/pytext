Installation
============

*PyText requires Python 3.6+*

PyText is available in the Python Package Index via

.. code-block:: console

  $ pip install pytext


The easiest way to get started on most systems is to create a `virtualenv`

.. code-block:: console

  $ python3 -m virtualenv venv
  $ source pytext/bin/activate
  (venv) $ pip install pytext

This will install a version of PyTorch depending on your system. See `PyTorch <https://pytorch.org>`_ for more information. If you are using MacOS or Windows, this likely will not include GPU support by default; if you are using Linux, you should automatically get a version of PyTorch compatible with CUDA 9.0.

If you need a different version of PyTorch, follow the instructions on the `PyTorch website <https://pytorch.org>`_ to install the appropriate version of PyTorch before installing PyText

OS Dependencies
===============

*if you're having issues getting things to run, these guides might help*

On MacOS
--------

Install `brew <https://brew.sh>`_, then run the command:

.. code-block:: console

  $ brew install cmake protobuf

On Windows
---------

Coming Soon!

On Linux
--------

For Ubuntu/Debian distros, you might need to run the following command:

.. code-block:: console

  $ sudo apt-get install protobuf-compiler libprotoc-dev


For rpm-based distros, you might need to run the following command:

.. code-block:: console

  $ sudo yum install protobuf-devel


Install From Source 
====================

.. code-block:: console

  $ git clone git@github.com:facebookresearch/pytext.git
  $ cd pytext
  $ source activation_venv
  (venv) $ pip install torch # go to https://pytorch.org for platform specific installs
  (venv) $ ./install_deps

Once that is installed, you can run the unit tests. We recommend using pytest as a runner.

.. code-block:: console

  (venv) $ pip install -U pytest
  (venv) $ pytest
  # If you want to measure test coverage, we recommend `pytest-cov`
  (venv) $ pip install -U pytest-cov
  (venv) $ pytest --cov=pytext

To resume development in an already checked-out repo:

.. code-block:: console

  $ cd pytext
  $ source activation_venv

To exit the virtual environment:

.. code-block:: console

   (venv) $ deactivate
