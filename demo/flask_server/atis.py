#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import re

import numpy as np
from caffe2.python import workspace
from caffe2.python.predictor import predictor_exporter


# Load Caffe2 model
predict_net = predictor_exporter.prepare_prediction_net(
    filename="atis_joint_model.c2", db_type="minidb"
)


# Pre-processing helper method
def tokenize(text):
    # Split by whitespace, force lowercase
    tokens = []
    token_ranges = []

    def add_token(text, start, end):
        token = text[start:end]
        if token:
            tokens.append(token)
            token_ranges.append((start, end))

    start = 0
    text = text
    for sep in re.finditer(r"\s+", text):
        add_token(text, start, sep.start())
        start = sep.end()
    add_token(text, start, len(text))

    if not tokens:
        # Add PAD_TOKEN in case of empty text
        tokens = ["<pad>"]
    tokens = list(map(str.lower, tokens))

    return tokens, token_ranges


# Run ATIS model
def predict(text):
    # Pre-process
    tokens, token_ranges = tokenize(text)

    # Make prediction
    workspace.blobs["tokens_vals_str:value"] = np.array([tokens], dtype=str)
    workspace.blobs["tokens_lens"] = np.array([len(tokens)], dtype=np.int_)
    workspace.RunNet(predict_net)
    labels_scores = [
        (str(blob), workspace.blobs[blob][0])
        for blob in predict_net.external_outputs
        if "word_scores" in str(blob)
    ]
    labels = list(zip(*labels_scores))[0]
    scores = list(zip(*labels_scores))[1]  # len(tokens) x 1

    # Post-processing (find city names)
    all_scores = np.concatenate(scores, axis=1)  # len(tokens) x len(labels)
    predicted_labels = np.argmax(all_scores, axis=1)  # len(tokens)

    city_token_ranges = []
    prev_label = ""
    for token_idx, label_idx in enumerate(predicted_labels):
        label = labels[label_idx]
        if "city_name" in label:
            if prev_label == label:
                city_token_ranges[-1] = (
                    city_token_ranges[-1][0],
                    token_ranges[token_idx][1],
                )
            else:
                city_token_ranges.append(token_ranges[token_idx])
        prev_label = label
    return city_token_ranges
