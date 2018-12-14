Data Parallel Distributed Training
===============================================

Distributed training enables one to easily parallelize computations across processes
and clusters of machines. To do so, it leverages messaging passing semantics allowing
each process to communicate data to any of the other processes.

PyText exploits ``DistributedDataParallel`` for synchronizing gradients and ``torch.multiprocessing``
to spawn multiple processes which each setup the distributed environment with `NCCL <https://developer.nvidia.com/nccl>`_ as
default backend, initialize the process group, and finally execute the given run function.
The module is replicated on each machine and each device (e.g every single process),
and each such replica handles a portion of the input partitioned by PyText's :class:`~DataHandler`.
For more on distributed training in PyTorch, refer to `Writing distributed applications with PyTorch
<https://pytorch.org/tutorials/intermediate/dist_tuto.html>`_.

In this tutorial, we will train a DocNN model on a single node with 8 GPUs using the SST dataset.

1. Requirement
--------------------

Distributed training is only available for GPUs, so you'll need GPU-equipped server or virtual machine to run this tutorial.

Notes:
 - This demo use a local temporary file for initializing the distributed processes group,
   which means it only works on a single node. Please make sure to set `distributed_world_size`
   less than or equal to the maximum available GPUs on the server.

 - For distributed training on clusters of machines, you can use a shared file accessible to
   all the hosts (ex: file:///mnt/nfs/sharedfile) or the TCP init method. More info on
   `distributed initialization
   <https://pytorch.org/docs/stable/distributed.html#initialization>`_.

 - In ``demo/configs/distributed_docnn.json``, set `distributed_world_size` to 1 to disable
   distributed training, and set `use_cuda_if_available` to `false` to disable training on GPU.


2. Fetch the dataset
--------------------

Download the `SST dataset (The Stanford Sentiment Treebank) <https://gluebenchmark.com/tasks>`_ to a local directory. We will refer to this as `base_dir` in the next section.

.. code-block:: console

  $ unzip SST-2.zip && cd SST-2
  $ sed 1d train.tsv | head -1000 > train_tiny.tsv
  $ sed 1d dev.tsv | head -100 > eval_tiny.tsv


3. Prepare configuration file
-----------------------------

Prepare the configuration file for training. A sample config file can be found in your PyText repository at ``demo/configs/distributed_docnn.json``. If you haven't set up PyText, please follow :doc:`installation`.

The two parameters that are used for distributed training are:

- `distributed_world_size`: total number of GPUs used for distributed training, e.g. if set to 40 with every server having 8 GPU, 5 servers will be fully used.
- `use_cuda_if_available`: set to `true` for training on GPUs.

For this tutorial, please change the following in the config file.

- Set `train_path` to `base_dir/train_tiny.tsv`.
- Set `eval_path` to `base_dir/eval_tiny.tsv`.
- Set `test_path` to `base_dir/eval_tiny.tsv`.

4. Train model with the downloaded dataset
------------------------------------------

Train the model using the command below

.. code-block:: console

  (pytext) $ pytext train < demo/configs/distributed_docnn.json
