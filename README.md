# Installing PyText

To get started, run the following commands in a terminal:

```
git clone git@github.com:facebookresearch/pytext.git
cd pytext

source activation_venv
./install_deps
./run_tests
```

To resume development in an already checked-out repo:

```
cd pytext
source activation_venv
```

To exit the virtual environment:

```
deactivate
```

Alternatively, if you don't want to run in a virtual env, you can install the dependencies globally with `sudo ./install_deps`.

For additional information, please read INSTALL.md

# Train your first classifier

For this first example, we'll use create a DocNN classifier that classifies the type of thing the user is asking for, using the examples in `tests/data/train_data_tiny.tsv`.

```
python3 pytext/main.py train < demo/configs/docnn.json
```

By default, the model is created in `/tmp/model.pt`

Now you can export your model as a caffe2 net:

```
pytext export < config.json
```

You can also run some predictions:

```
pytext --config-file config.json predict <<< '{"raw_text": "create an alarm for 1:30 pm"}'
```
