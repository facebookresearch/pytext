#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json

import atis
from flask import Flask, request


app = Flask(__name__)


@app.route("/")
def predict():
    return json.dumps(atis.predict(request.args.get("text", "")))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
