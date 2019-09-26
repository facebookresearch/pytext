Execute your first model
=================================

In :doc:`train_your_first_model`, we learnt how to train a small, simple model. We can continue this tutorial with that model here. This procedure can be used for any pytext model by supplying the matching config. For example, the much more powerful model from :doc:`atis_tutorial` can be executed using this same procedure.

Evaluate the model
--------------------

We want to run the model on our test dataset and see how well it performs. Some results have been abbreviated for clarity.

.. code-block:: console

    (pytext) $ pytext test < demo/configs/docnn.json

    Stage.TEST
    loss: 2.059336
    Accuracy: 20.00

    Macro P/R/F1 Scores:
        Label                       Precision   Recall      F1          Support

        reminder/set_reminder       25.00       100.00      40.00       1
        alarm/time_left_on_alarm    0.00        0.00        0.00        1
        alarm/show_alarms           0.00        0.00        0.00        1
        alarm/set_alarm             0.00        0.00        0.00        2
        Overall macro scores        6.25        25.00       10.00

    Soft Metrics:
        Label       Average precision
        alarm/set_alarm 50.00
        alarm/time_left_on_alarm    20.00
        reminder/set_reminder   25.00
        alarm/show_alarms   20.00
        weather/find    nan
        alarm/modify_alarm  nan
        alarm/snooze_alarm  nan
        reminder/show_reminders nan
        Label       Recall at precision 0.2
        alarm/set_alarm 100.00
        Label       Recall at precision 0.4
        alarm/set_alarm 100.00
        Label       Recall at precision 0.6
        alarm/set_alarm 0.00
        Label       Recall at precision 0.8
        alarm/set_alarm 0.00
        Label       Recall at precision 0.9
        alarm/set_alarm 0.00
        Label       Recall at precision 0.2
        alarm/time_left_on_alarm    100.00
        Label       Recall at precision 0.4
        alarm/time_left_on_alarm    0.00
        Label       Recall at precision 0.6
        alarm/time_left_on_alarm    0.00
    ... [snip]
        reminder/show_reminders 0.00
        Label       Recall at precision 0.6
        reminder/show_reminders 0.00
        Label       Recall at precision 0.8
        reminder/show_reminders 0.00
        Label       Recall at precision 0.9
        reminder/show_reminders 0.00


Export the model
-------------------

When you save a PyTorch model, the snapshot uses `pickle` for serialization. This means that simple code changes (e.g. a word embedding update) can cause backward incompatibilities with your deployed model. To combat this, you can export your model into the `Caffe2 <https://caffe2.ai/>`_ format using in-built `ONNX <https://onnx.ai/>`_ integration. The exported Caffe2 model would have the same behavior regardless of changes in PyText or in your development code.

Exporting a model is pretty simple:

.. code-block:: console

    (pytext) $ pytext export --help
    Usage: pytext export [OPTIONS]

      Convert a pytext model snapshot to a caffe2 model.

    Options:
      --model TEXT        the pytext snapshot model file to load
      --output-path TEXT  where to save the exported model
      --help              Show this message and exit.

You can also pass in a configuration to infer some of these options. In this case let's do that because depending on how you're following along your snapshot might be in different places!

.. code-block:: console

    (pytext) $ pytext export --output-path exported_model.c2 < demo/configs/docnn.json
    ...[snip]
    Saving caffe2 model to: exported_model.c2

This file now contains all of the information needed to run your model.

There's an important distinction between what a model does and what happens before/after the model is called, i.e. the preprocessing and postprocessing steps. PyText strives to do as little preprocessing as possible, but one step that is very often needed is tokenization of the input text. This will happen automatically with our prediction interface, and if this behavior ever changes, we'll make sure that old models are still supported. The model file you export will always work, and you don't necessarily need PyText to use it! Depending on your use case you can implement preprocessing yourself and call the model directly, but that's outside the scope of this tutorial.

Make a simple app
-------------------

Let's put this all into practice! How might we make a simple web app that loads an exported model and does something meaningful with it?

To run the following code, you should

.. code-block:: console

    (pytext) $ pip install flask

Then we implement a minimal `Flask <http://flask.pocoo.org/>`_ web server.

.. code-block:: python

    import sys
    import flask
    import pytext

    config_file = sys.argv[1]
    model_file = sys.argv[2]

    config = pytext.load_config(config_file)
    predictor = pytext.create_predictor(config, model_file)

    app = flask.Flask(__name__)

    @app.route('/get_flight_info', methods=['GET', 'POST'])
    def get_flight_info():
        text = flask.request.data.decode()

        # Pass the inputs to PyText's prediction API
        result = predictor({"text": text})

        # Results is a list of output blob names and their scores.
        # The blob names are different for joint models vs doc models
        # Since this tutorial is for both, let's check which one we should look at.
        doc_label_scores_prefix = (
            'scores:' if any(r.startswith('scores:') for r in result)
            else 'doc_scores:'
        )

        # For now let's just output the top document label!
        best_doc_label = max(
            (label for label in result if label.startswith(doc_label_scores_prefix)),
            key=lambda label: result[label][0],
        # Strip the doc label prefix here
        )[len(doc_label_scores_prefix):]

        return flask.jsonify({"question": f"Are you asking about {best_doc_label}?"})

    app.run(host='0.0.0.0', port='8080', debug=True)


Execute the app

.. code-block:: console

    (pytext) $ python flask_app.py demo/configs/docnn.json exported_model.c2
    * Serving Flask app "flask_app" (lazy loading)
    * Environment: production
      WARNING: Do not use the development server in a production environment.
      Use a production WSGI server instead.
    * Debug mode: on

Then in a separate terminal window

.. code-block:: console

    $ function ask_about() { curl http://localhost:8080/get_flight_info -H "Content-Type: text/plain" -d "$1" }

    $ ask_about 'I am looking for flights from San Francisco to Minneapolis'
    {
      "question": "Are you asking about flight?"
    }

    $ ask_about 'How much does a trip to NY cost?'
    {
      "question": "Are you asking about airfare?"
    }

    $ ask_about "Which airport should I go to?"
    {
      "question": "Are you asking about airport?"
    }
