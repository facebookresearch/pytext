#!/usr/bin/env python3

from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from pytext.config import PyTextConfig, config_from_json, config_to_json
from pytext.config.component import create_data_handler, create_model
from pytext.data.data_handler import DataHandler


def save(
    save_path: str, config: PyTextConfig, model: nn.Module, data_handler: DataHandler
) -> None:
    model.save_modules()
    predictor_state = OrderedDict(
        [
            ("data_state", data_handler.metadata_to_save()),
            ("config_json", config_to_json(PyTextConfig, config)),
            ("model_state", model.state_dict()),
        ]
    )  # type: OrderedDict
    torch.save(predictor_state, save_path)


def load(load_path: str) -> Tuple[PyTextConfig, nn.Module, DataHandler]:

    predictor_state = torch.load(load_path, map_location=lambda storage, loc: storage)

    metadata = predictor_state["data_state"]
    model_state_dict = predictor_state["model_state"]
    config = config_from_json(PyTextConfig, predictor_state["config_json"])
    jobspec = config.jobspec
    data_handler = create_data_handler(
        jobspec.data_handler, jobspec.features, jobspec.labels
    )
    data_handler.load_metadata(metadata)
    model = create_model(jobspec.model, jobspec.features, data_handler.metadata)
    model.load_state_dict(model_state_dict)
    return config, model, data_handler
