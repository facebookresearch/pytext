#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import csv
import json
import logging
import threading
from typing import Any, Dict, Tuple

import requests
from flask import Blueprint, Flask, jsonify, render_template, request
from flask_cors import CORS


# a lock for writing to the output
# data file
DATA_FILE_LOCK = threading.Lock()


def get_args() -> argparse.ArgumentParser:
    """
    Return CLI configuration for running server
    """
    parser = argparse.ArgumentParser(description="Run the html hosting server")
    parser.add_argument(
        "--port", default=5000, type=int, help="the port to launch server on"
    )

    parser.add_argument("--modelserver", default="http://localhost:8080")

    parser.add_argument(
        "--debug", action="store_true", help="Launch debug version of server"
    )

    parser.add_argument(
        "--datafile", default="data.csv", help="the file to write data to"
    )

    parser.add_argument(
        "--console_address",
        default="http://localhost:5000",
        help="the location of the console server",
    )
    return parser


logger = logging.getLogger(__name__)


def get_key_from_data(data, key):
    if key in data:
        return data[key]
    return None


def create_app(config_filename: str):
    api_bp = Blueprint("api", __name__)
    app = Flask(__name__, static_url_path="/static")
    CORS(app, resources=r"/api/*")

    app.register_blueprint(api_bp, url_prefix="/api")

    return app


def setup_app(app: Flask, args: object):
    @app.route("/api/model/", methods=["GET"])
    def get_predictions():
        query = get_key_from_data(request.args, "query")
        payload = {"text": str(query)}
        r = requests.post(args.modelserver, json=payload)
        response = json.loads(r.text)
        intent_scores = response["intent_ranking"]

        intent_scores = list(filter(lambda inp: inp, intent_scores))

        def convert_to_tuple(intent_score: Dict[str, Any]) -> Tuple[str, float]:
            intent_name = intent_score["name"]
            intent_score = intent_score["confidence"]
            return intent_name, intent_score

        intent_scores = list(map(convert_to_tuple, intent_scores))
        intent_scores = sorted(intent_scores, reverse=True, key=lambda tup: tup[1])
        return (
            jsonify(
                {
                    "query": query,
                    "prediction": intent_scores[0],
                    "raw_scores": intent_scores,
                }
            ),
            200,
        )

    @app.route("/api/add_data/", methods=["GET"])
    def upload_data():
        data_point = get_key_from_data(request.args, "data_point")
        query, label = data_point.split(",")
        with DATA_FILE_LOCK:
            with open(args.datafile, "a+") as csv_file:
                csvwriter = csv.writer(csv_file, delimiter="\t")
                csvwriter.writerow([query, label])

        return jsonify({"query": query, "label": label}), 200

    @app.route("/", methods=["GET"])
    def root():
        return render_template("index.html", console_address=args.console_address)


def main():
    args = get_args().parse_args()

    app = create_app("config")
    setup_app(app, args)
    try:
        app.run(host="0.0.0.0", debug=args.debug, port=args.port)
    except KeyboardInterrupt:
        print("Received Keyboard interrupt, exiting")


if __name__ == "__main__":
    main()
