Deploying your first model (part 2)
==================================

In _`train_your_first_model` we trained a test model. You can continue this document with that model, but the results won't be very compelling because the model hasn't learned anything. I'm going to continue with the model trained in _`atis_tutorial`, but you can follow along just as easily from the test model. To make it easier to write these commands for either, I recommend setting a shell variable to the config you're using in either case.

If you want to quickly run through this with the test model trained in part 1, do

.. code-block:: console

    (pytext) $ CONFIG=demo/configs/docnn.json

If you're following along with the atis tutorial, use this config

.. code-block:: console

    (pytext) $ CONFIG=demo/atis_joint_model/atis_joint_config.json

Evaluating the model
--------------------

We want to run our model on our test dataset and see how well it performs. I'm abbreviating some of the results for clarity.

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

Exporting the model
-------------------

Now we want to export the model to a PyTorch caffe2 serialized model. There are other ways to deploy models depending on your needs, but this one has some important advantages. The biggest thing it gives you is backwards compatibility. The caffe2 serialized format is standardized and has good guarantees about backwards compatibility. That means once you export your model, it's saved in a way such that PyTorch will know how to execute it correctly forever. No matter what changes you make to your code, or what changes are made to PyText, this exported model will continue to work properly in a production environment. By contrast for instance, the snapshots that are saved for evaluation before exporting use pickle, which means that various types of simple code changes, or say if you update the word embeddings you're using, can make your model break or even worse, silently behave incorrectly.

Exporting a model is pretty simple.

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

This model file now contains all of the information needed to run your model.

There's an important distinction that needs to be made between what a model does and what happens before/after a model is called, ie. the preprocessing and postprocessing steps. PyText tries to do as little preprocessing as possible, but one thing that it always needs to do (for word-level models at least) is to tokenize the input text. This will happen automatically with our prediction interface, and if this behavior ever changes, we'll make sure that old models are still supported. The model file you export will always work, and you don't necessarily need PyText to use it! Depending on your use case you can implement preprocessing yourself and call the model directly with torch, but that's outside the scope of this tutorial.

Making a simple app
-------------------

Let's put this all into practice! How might we make a simple web app that loads an exported model and does something meaningful with it?

To run the following code, you should

.. code-block:: console

    (pytext) $ pip install flask

I have this code in a file called flask_app.py.

::

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

Then in a separate terminal


.. code-block:: console

    $ function ask_about() { curl http://localhost:8080/get_flight_info -H "Content-Type: text/plain" -d "$1" }

    $ ask_about 'I am looking for flights from San Francisco to Minneapolis
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
