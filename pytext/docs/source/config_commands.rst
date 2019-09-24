Config Commands
===============

This page explains the usage of the commands ``help-config`` to explore PyText components, and ``gen-default-config`` to create a config file with custom components and parameters.

Exploring Config Options
------------------------

You can explore PyText Components with the command ``help-config``. This will print the documentation of the component, its full module name, its base class, as well as the list of its config parameters, their type and their default value.

.. code-block:: console

    $ pytext help-config LMTask
    === pytext.task.tasks.LMTask (NewTask) ===
        data = Data
        exporter = null
        features = FeatureConfig
        featurizer = SimpleFeaturizer
        metric_reporter: LanguageModelMetricReporter = LanguageModelMetricReporter
        model: LMLSTM = LMLSTM
        trainer = TaskTrainer

You can drill down to the component you're interested in. For example, if you want to know more about the model :class:`~LMLSTM`, you can use the same command. Notice how PyText lists the possible values for `Union` types (for example with `representation` below.)

.. code-block:: console

    $ pytext help-config LMLSTM
    === pytext.models.language_models.lmlstm.LMLSTM (BaseModel) ===
    """
    `LMLSTM` implements a word-level language model that uses LSTMs to
        represent the document.
    """
        ModelInput = LMLSTM.Config.ModelInput
        caffe2_format: (ExporterType)
             PREDICTOR (default)
             INIT_PREDICT
        decoder: (one of)
             None
             MLPDecoder (default)
        embedding: WordFeatConfig = WordEmbedding
        inputs: LMLSTM.Config.ModelInput = ModelInput
        output_layer: LMOutputLayer = LMOutputLayer
        representation: (one of)
             DeepCNNRepresentation
             BiLSTM (default)
        stateful: bool
        tied_weights: bool


PyText internally registers all the component classes, so we can look up and find any component using the class name or their aliases. For example somewhere in PyText we have :class:`~import DeepCNNRepresentation as CNN`, so we would normally look up :class:`~DeepCNNRepresentation`, but if we know that this class has an alias we can look up :class:`~CNN` instead, and print the information about this class:

.. code-block:: console

    $ pytext help-config CNN
    === pytext.models.representations.deepcnn.DeepCNNRepresentation (RepresentationBase) ===
    """
    `DeepCNNRepresentation` implements CNN representation layer
        preceded by a dropout layer. CNN representation layer is based on the encoder
        in the architecture proposed by Gehring et. al. in Convolutional Sequence to
        Sequence Learning.

        Args:
            config (Config): Configuration object of type DeepCNNRepresentation.Config.
            embed_dim (int): The number of expected features in the input.
    """
        cnn: CNNParams = CNNParams
        dropout: float = 0.3


Creating a Config File
----------------------

The command ``gen-default-config`` creates a json config files for a given :class:`~Task` using the default value for all the parameters. You must specify the class name of the :class:`~Task`. The json config will be printed in the terminal, so you need to send it to a file using of your choice (for example ``my_config.json``) to be able to `edit it and use it <config_files.html>`_.

.. code-block:: console

    $ pytext gen-default-config LMTask > my_config.json
    INFO - Applying task option: LMTask
    ...


In the ``help-config LMLSTM`` above, we see that `representation` is by default :class:`~BiLSTM`, but could also be :class:`~DeepCNNRepresentation`. (This can be because the type is declared as a `Union` of valid alternatives, or because the type is a base class.) Those two classes will have different parameters, so we can't just edit the `my_config.json` and replace the class name.

We can specify which components to use by adding any number of class names to the command. Let's create this config, and we'll use add :class:`~DeepCNNRepresentation` to our command. ``gen-default-config`` will look up this class name and find that it is a suitable `representation` component for the :class:`~LMLSTM` model in our :class:`~LMTask`.

.. code-block:: console

    $ pytext gen-default-config LMTask DeepCNNRepresentation > my_config.json
    INFO - Applying task option: LMTask
    INFO - Applying class option: task->model->representation = CNN
    ...


This also works with parameters which are not component class names. You can specify the parameter name and its value, and ``gen-default-config`` will automatically apply this parameter to the right component.

.. code-block:: console

    $ pytext gen-default-config LMTask epochs=200
    INFO - Applying task option: LMTask
    INFO - Applying parameter option to task.trainer.epochs : epochs=200
    ...


Sometimes the same parameter name is used by multiple components. In this case PyText prints the list of those parameters with their full config path. You can then simply use the last part of the path that is enough to differentiate them and pick the one you want. In the next example, we omit the prefix `task.model.` because we don't need it to find where to apply our parameter `representation.dropout`.

.. code-block:: console

    $ pytext gen-default-config LMTask dropout=0.7 > my_config.json
    INFO - Applying task option: LMTask
    ...
    Exception: Multiple possibilities for dropout=0.7: task.model.representation.dropout, task.model.decoder.dropout

    $ pytext gen-default-config LMTask representation.dropout=0.7 > my_config.json
    INFO - Applying task option: LMTask
    INFO - Applying parameter option to task.model.representation.dropout : representation.dropout=0.7
    ...


You can add any number and combination of those parameters. Please note that they will be applied in order, so if you want to change a component class and some of its parameters, you must specify the parameters in this order (component first, then parameters). If you don't do that, your parameters changes will be ignored. For example, changing `representation.dropout` first, then overriding the representation component will replace the default representation with a new :class:`~CNN` component with all the parameter using the default value.

Look at this bad example: you can verify that the representation dropout is 0.3 (the default value for :class:`~CNN`) and not 0.7 as we specified, because CNN was applied after and replaced the component that had its dropout modified first.

.. code-block:: console

    $ pytext gen-default-config LMTask representation.dropout=0.7 CNN > my_config.json
    INFO - Applying task option: LMTask
    INFO - Applying parameter option to task.model.representation.dropout : representation.dropout=0.7
    INFO - Applying class option: task->model->representation = CNN
    ...


Now let's combine everything:

.. code-block:: console

    $ pytext gen-default-config LMTask BlockShardedTSVDataSource CNN dilated=True epochs=200 representation.dropout=0.7 > my_config.json
    INFO - Applying task option: LMTask
    INFO - Applying class option: task->data->source = BlockShardedTSVDataSource
    INFO - Applying class option: task->model->representation = CNN
    INFO - Applying parameter option to task.model.representation.cnn.dilated : dilated=True
    INFO - Applying parameter option to task.trainer.epochs : epochs=200
    INFO - Applying parameter option to task.model.representation.dropout : representation.dropout=0.2
    ...


Updating a Config File
----------------------

When there's a new release of PyText, some component parameters might change because of bug fixes or new features. While PyText has `config_adapters` that can internally transform old configs to map them to the latest components, it is sometimes useful to update your config file to the current version. This can be done with the command ``update-config``:

.. code-block:: console

    $ pytext update-config < my_config_old.json > my_config_new.json

