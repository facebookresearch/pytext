#!/usr/bin/env python3

import argparse
from pytext.config import PyTextConfig, config_from_json
from pytext.common.registry import Registry, JOB_SPEC
import os
import json


def parse_json():
    """
    Parse json obj from command line parameter or parameter file
    """
    parser = argparse.ArgumentParser(description="PyText modeling framework")
    parser.add_argument("-parameters-file", default="", type=str)
    parser.add_argument("-parameters-json", default="", type=str)
    args = parser.parse_args()
    if os.path.isfile(args.parameters_file):
        with open(args.parameters_file) as params_f:
            params_json = json.load(params_f)
    else:
        params_json = json.loads(args.parameters_json)

    return params_json


def parse_pytext_config(config_json):
    """
    Parse json object into PyTextConfig object
    """
    PyTextConfig._field_types["jobspec"].__args__ = tuple(Registry.values(JOB_SPEC))
    return config_from_json(PyTextConfig, config_json)


def parse_config():
    """
    Parse PyTextConfig object from parameter string or parameter file
    """
    params_json = parse_json()
    # TODO T32608471 should assume the entire json is PyTextConfig later, right
    # now we're matching the file format for pytext trainer.py inside fbl
    pytext_config = parse_pytext_config(params_json["config"])
    return pytext_config
