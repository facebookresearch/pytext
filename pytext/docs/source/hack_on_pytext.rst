Hack on PyText
==============

To get started, run the following commands in a terminal

.. code-block:: console

		$ git clone git@github.com:facebookresearch/pytext.git
		$ cd pytext

		$ source activation_venv
		(pytext) $ ./install_deps

Next, install the PyTorch via pip using the `official instructions <https://pytorch.org>`_, making sure to match your OS and GPU environment.
Once you're done with that, you can run the tests. We recommend using `pytest`

.. code-block:: console

		(pytext) $ pip install -U pytest
		(pytext) $ pytest

To resume development in an already checked-out repo

.. code-block:: console

		$ cd pytext
		$ source activation_venv

To exit the virtual environment

.. code-block:: console

		(pytext) $ deactivate


Alternatively, if you don't want to run in a `virtualenv`, you can install the dependencies globally with `sudo ./install_deps`.

For additional information, please read `INSTALL.md`
