Execute your first model
=================================

In :doc:`train_your_first_model`, we learnt how to train a very small, simple model. You can easily continue this tutorial with that model, but the results won't be very compelling. Here we'll use the much more powerful model from :doc:`atis_tutorial`. To make the commands below compatible with either, we recommend setting a shell variable to the config you're using.

If you want to quickly run through this with the test model trained in the previous tutorial, do

.. code-block:: console

    (pytext) $ CONFIG=demo/configs/docnn.json

If you're following along with the ATIS tutorial, use this config

.. code-block:: console

    (pytext) $ CONFIG=demo/atis_joint_model/atis_joint_config.json

Evaluate the model
--------------------

We want to run the model on our test dataset and see how well it performs. Some results have been abbreviated for clarity.

.. code-block:: console

    (pytext) $ pytext test < "$CONFIG"

    Intent Metrics
            Per label scores                                Precision       Recall          F1              Support

            flight                                          98.27           99.05           98.66           632
            flight_no                                       88.89           100.00          94.12           8
            abbreviation                                    94.29           100.00          97.06           33
            ground_service                                  94.74           100.00          97.30           36
            airfare                                         92.31           100.00          96.00           48
            airline                                         97.44           100.00          98.70           38
    ...[snip]
           ... [snip]

            Overall micro scores                            96.64           96.64           96.64           893
            Overall macro scores                            73.03           69.55           69.14


    Slot Metrics
            Per label scores                                Precision       Recall          F1              Support

            toloc.city_name                                 97.12           98.88           97.99           715
            fromloc.city_name                               98.46           99.72           99.08           703
            fare_basis_code                                 88.89           94.12           91.43           17
    ...[snip]

            Overall micro scores                            94.88           95.13           95.01           2260
            Overall macro scores                            70.13           71.12           69.09

Not bad!

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

    (pytext) $ pytext export --output-path exported_model.c2 < "$CONFIG"
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
        result = predictor({"raw_text": text})

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

    (pytext) $ python flask_app.py "$CONFIG" exported_model.c2
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
