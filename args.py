#!/usr/bin/env python3

import argparse
import json
import os

from pytext.config import PyTextConfig, TestConfig, config_from_json


TRAIN_MODE = "train"
TEST_MODE = "test"


def setup_parser():
    parser = argparse.ArgumentParser(description="PyText modeling framework")
    subparsers = parser.add_subparsers(dest="mode")
    parser_train = subparsers.add_parser(TRAIN_MODE, help="train help")
    parser_test = subparsers.add_parser(TEST_MODE, help="test help")
    parser_train.add_argument("-parameters-file", default="", type=str)
    parser_train.add_argument("-parameters-json", default="", type=str)
    parser_test.add_argument("-parameters-file", default="", type=str)
    parser_test.add_argument("-parameters-json", default="", type=str)
    return parser


def parse_json():
    """
    Parse json obj from command line parameter or parameter file
    """
    parser = setup_parser()
    args = parser.parse_args()
    if args.mode == TRAIN_MODE:
        pass
    elif args.mode == TEST_MODE:
        pass
    else:
        raise Exception(f"unknow mode {args.mode}")

    if os.path.isfile(args.parameters_file):
        with open(args.parameters_file) as params_f:
            params_json = json.load(params_f)
    else:
        params_json = json.loads(args.parameters_json)

    return args.mode, params_json


def parse_config():
    """
    Parse PyTextConfig object from parameter string or parameter file
    """
    mode, params_json = parse_json()
    confg_cls = {TRAIN_MODE: PyTextConfig, TEST_MODE: TestConfig}[mode]
    # TODO T32608471 should assume the entire json is PyTextConfig later, right
    # now we're matching the file format for pytext trainer.py inside fbl
    config = config_from_json(confg_cls, params_json["config"])
    return mode, config
