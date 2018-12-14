Installation
============

*PyText requires Python 3.6+*

PyText is available in the Python Package Index via

.. code-block:: console

  $ pip install pytext-nlp


The easiest way to get started on most systems is to create a `virtualenv`

.. code-block:: console

  $ python3 -m venv pytext_venv
  $ source pytext_venv/bin/activate
  (pytext_venv) $ pip install pytext-nlp

This will install a version of PyTorch depending on your system. See `PyTorch <https://pytorch.org>`_ for more information. If you are using MacOS or Windows, this likely will not include GPU support by default; if you are using Linux, you should automatically get a version of PyTorch compatible with CUDA 9.0.

If you need a different version of PyTorch, follow the instructions on the `PyTorch website <https://pytorch.org>`_ to install the appropriate version of PyTorch before installing PyText



OS Dependencies
---------------

*if you're having issues getting things to run, these guides might help*

On MacOS
^^^^^^^^^

Install `brew <https://brew.sh>`_, then run the command:

.. code-block:: console

  $ brew install cmake protobuf

On Windows
^^^^^^^^^^^

Coming Soon!

On Linux
^^^^^^^^^

For Ubuntu/Debian distros, you might need to run the following command:

.. code-block:: console

  $ sudo apt-get install protobuf-compiler libprotoc-dev


For rpm-based distros, you might need to run the following command:

.. code-block:: console

  $ sudo yum install protobuf-devel


Install From Source
--------------------

.. code-block:: console

  $ git clone git@github.com:facebookresearch/pytext.git
  $ cd pytext
  $ source activation_venv
  (pytext_venv) $ pip install torch # go to https://pytorch.org for platform specific installs
  (pytext_venv) $ ./install_deps

Once that is installed, you can run the unit tests. We recommend using pytest as a runner.

.. code-block:: console

  (pytext_venv) $ pip install -U pytest
  (pytext_venv) $ pytest
  # If you want to measure test coverage, we recommend `pytest-cov`
  (pytext_venv) $ pip install -U pytest-cov
  (pytext_venv) $ pytest --cov=pytext

To resume development in an already checked-out repo:

.. code-block:: console

  $ cd pytext
  $ source activation_venv

To exit the virtual environment:

.. code-block:: console

   (pytext_venv) $ deactivate


Cloud VM Setup
---------------

This guide will cover all the setup work you have to do in order to be able to easily install PyText on a cloud VM
.
*Note that while these instructions worked when they were written, they may become incorrect or out of date. If they do, please send us a Pull Request!*

After following these instructions, you should be good to either follow the `Installation`_ instructions or the `Install From Source`_ instructions

Amazon Web Services
^^^^^^^^^^^^^^^^^^^^
**Coming Soon**

Google Cloud Engine
^^^^^^^^^^^^^^^^^^^^

*If you have problems launching your VM, make sure you have a non-zero gpu quota,* `click here to learn about quotas <https://cloud.google.com/compute/quotas#requesting_additional_quota>`_

This guide uses `Google's Deep Learning VM <https://console.cloud.google.com/marketplace/details/click-to-deploy-images/deeplearning>`_ as a base.

**Setting Up Your VM**

* Click "Launch on Compute Engine"
* Configure the VM:

  * The default 2CPU K80 setup is fine for most tutorials, if you need more, change it here.
  * For Framework, select one of the Base images, rather than one with a framework pre-installed. Note which version of CUDA you choose for later.
  * When you're ready, click "Deploy"
  * When your VM is done loading, you can SSH into it from the GCE Console

* Install Python 3.6 (based on `this RoseHosting blog post <https://www.rosehosting.com/blog/how-to-install-python-3-6-4-on-debian-9/>`_ ):

  * ``$ sudo nano /etc/apt/sources.list``
  * add ``deb http://ftp.de.debian.org/debian testing main`` to the list
  * ``$ echo 'APT::Default-Release "stable";' | sudo tee -a /etc/apt/apt.conf.d/00local``
  * ``$ sudo apt-get update``
  * ``$ sudo apt-get -t testing install python3.6``
  * ``$ sudo apt-get install python3.6-venv protobuf-compiler libprotoc-dev``


Microsoft Azure
^^^^^^^^^^^^^^^^^

This guide uses the Azure Ubuntu Server 18.04 LTS image as a base

**Setting Up Your VM**

* From the Azure Dashboard, select "Virtual Machines" and then click "add"
* Give your VM a name and select the region you want it in, keeping in mind that GPU servers are not present in all regions
* For this tutorial, you should select "Ubuntu Server 18.04 LTS" as your image
* Click "Change size" in order to select a GPU server.

  * Note that the default filters won't show GPU servers, we recommend clearing all filters except "family" and setting "family" to GPU
  * For this tutorial, we will use the NC6 VM Size, but this should work on the larger and faster VMs as well
* Make sure you set up SSH access, we recommend using a public key rather than a password.
  * don't forget to "allow selected ports" and select SSH

* install Nvidia driver and CUDA, (based on  https://askubuntu.com/a/1036265)

  * ``sudo add-apt-repository ppa:graphics-drivers/ppa``
  * ``sudo apt update``
  * ``sudo apt-get install ubuntu-drivers-common``
  * ``sudo ubuntu-drivers autoinstall``
  * reboot: ``sudo shutdown -r now``
  * ``sudo apt install nvidia-cuda-toolkit gcc-6``

* install OS dependencies: ``sudo apt-get install python3-venv protobuf-compiler libprotoc-dev``
