Hack on PyText
==============

To get started, run the following commands in a terminal::

		$ git clone git@github.com:facebookresearch/pytext.git
		$ cd pytext

		$ source activation_venv
		(pytext) $ ./install_deps

Next Install the PyTorch using Pip using the instructions on https://pytorch.org, making sure to match your OS and GPU situation
Once you're done with that, you can run the tests with::
  
		(pytext) $ ./run_tests

To resume development in an already checked-out repo::

		$ cd pytext
		$ source activation_venv

To exit the virtual environment::

		(pytext) $ deactivate


Alternatively, if you don't want to run in a virtual env, you can install the dependencies globally with `sudo ./install_deps`.

For additional information, please read INSTALL.md
