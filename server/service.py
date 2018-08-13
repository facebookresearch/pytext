#!/usr/bin/env python3

from pytext.server.model_loader import ModelLoader
from pytext.server.prediction_service import PytextPredictionService
from configerator.client import ConfigeratorClient
from messenger.assistant.pytext.server.config.ttypes import ServerConfig
from fblearner_predictor.model.definitions.ttypes import StructuredPrediction
from fblearner_predictor.prediction_service.ttypes import (
    PredictionException,
    PredictionRequest,
    PredictionResponse,
)
from libfb.py.controller.base import BaseController
from libfb.py.controller.tasks import PeriodicTask

import logging
import traceback
import pandas as pd
from typing import Dict, List, Any
from .util import convert_to_model_predictions

logging.basicConfig(level=logging.INFO)


class PytextModelUnloader(PeriodicTask):
    # Check and unload any models not accessed for a period of time every hour
    INTERVAL = 3600

    def initTask(self):
        super(PytextModelUnloader, self).initTask()

    def execute(self):
        try:
            if self.controller is not None and self.controller.model_loader is not None:
                self.controller.model_loader.unload_models()
        except Exception as e:
            if self.controller is not None and self.controller.logger is not None:
                self.controller.logger.error(
                    "Exception occurred while attempting to unload "
                    "outdated models: {}.".format(str(e))
                )


class PytextPredictionServiceHandler(BaseController, PytextPredictionService.Iface):
    SERVICE = PytextPredictionService

    TASKS = [PytextModelUnloader]

    @classmethod
    def makeArgumentParser(cls):
        ap = super().makeArgumentParser()
        ap.add_argument("--server-config", type=str, help="Path to server config file.")
        return ap

    def initService(self) -> None:
        self.logger = logging.getLogger(__name__)
        server_config_path = self.getOption("server-config")
        configerator_client = ConfigeratorClient()
        if not configerator_client.does_config_exist(server_config_path):
            raise Exception(
                """Specified server config path {} does not exist.""".format(
                    server_config_path
                )
            )
        config_thrift = configerator_client.get_config_contents_as_thrift(
            server_config_path, ServerConfig
        )
        self.model_loader = ModelLoader(config_thrift.unload_models_older_than_n_days)
        for model_id in config_thrift.models_to_preload:
            try:
                self.model_loader.load_model(model_id)
            except Exception as e:
                self.logger.warning(
                    "Pytext model ID: {} failed to preload. Skipping...".format(
                        model_id
                    )
                )
                continue
            self.logger.info(
                "Preloaded Pytext model ID: {} into memory.".format(model_id)
            )

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        model_ids = request.model_ids_hdfs
        if not model_ids:
            raise PredictionException(message="Specified model_ids must be non-empty")
        structured_predictions_hdfs = []
        for model_id in model_ids:
            try:
                model_handle = self.model_loader.load_model(
                    model_id.model, model_id.snapshot
                )
            except Exception as e:
                raise PredictionException(
                    message="""Could not load specified Pytext model [{}]
                    (common reason: Specified model is not a Pytext model).
                    Exception: {}. Stack trace: {}""".format(
                        model_id, str(e), traceback.format_exc()
                    )
                )
            try:
                # Convert the examples list to a flat list of named features
                df = pd.DataFrame(
                    [
                        {
                            feat_n: feat_v.value
                            for feat_n, feat_v in ex.named_features.items()
                        }
                        for ex in request.examples
                    ]
                )
                model_predictions = convert_to_model_predictions(
                    model_handle.predict(df)
                )
            except Exception as e:
                raise PredictionException(
                    message="""Exception occurred while running Pytext model:
                    {}. Stack trace: {}""".format(
                        str(e), traceback.format_exc()
                    )
                )
            structured_predictions_hdfs.append(
                StructuredPrediction(
                    model_info=model_id, model_predictions=model_predictions
                )
            )
        return PredictionResponse(
            structured_predictions_hdfs=structured_predictions_hdfs
        )


if __name__ == "__main__":
    PytextPredictionServiceHandler.initFromCLI()
