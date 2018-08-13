#!/usr/bin/env python3

from pytext.serialize import load
from pytext.registry import Registry
from fblearner.predictor.publishing.publisher import get_model_hdfs_path
from typing import Any, Optional

import fbcommon._drhdfs as hdfs
import gc
import logging
import os
import tempfile
import time

logging.basicConfig(level=logging.INFO)


SECS_PER_DAY = 24 * 60 * 60


class ModelLoader(object):
    def __init__(self, unload_models_older_than_n_days):
        self.model_handles = {}
        self.logger = logging.getLogger(__name__)
        self.unload_models_older_than_n_days = unload_models_older_than_n_days

    def _copy_hdfs_file(self, src: str, dst: str) -> None:
        self.logger.info('Copying from hdfs path: "{}" to "{}"'.format(src, dst))
        hdfs.download(src, dst)

    def _hdfs_path_of_model(self, model_id: int, snapshot_id: int) -> Optional[str]:
        hdfs_path_possibly_non_existent = get_model_hdfs_path(model_id, snapshot_id)
        if hdfs.exists(hdfs_path_possibly_non_existent):
            return hdfs_path_possibly_non_existent
        else:
            self.logger.error(
                'Model at HDFS path: "{}" not found'.format(
                    hdfs_path_possibly_non_existent
                )
            )
            return None

    def _download_model(self, source_path: str, dest_path: str) -> None:
        try:
            if not os.path.exists(dest_path):
                self._copy_hdfs_file(source_path, dest_path)
                # Check if the file exists. If not, HDFS have failed silently.
                if not os.path.exists(dest_path):
                    raise Exception(
                        "Failed to download model file from {} to {}".format(
                            source_path, dest_path
                        )
                    )
            else:
                self.logger.info(
                    "Skipping file downloading since file is already in "
                    "base_path: {}".format(dest_path)
                )
        except Exception:
            self.logger.exception("Failed to download from HDFS!")
            os.remove(dest_path)
            raise

    def unload_models(self) -> None:
        for model_id in self.model_handles.keys():
            (model_handle, last_access_timestamp) = self.model_handles[model_id]
            if (
                last_access_timestamp
                + self.unload_models_older_than_n_days * SECS_PER_DAY
                < time.time()
            ):
                del self.model_handles[model_id]
        gc.collect()

    def load_model(self, model_id: int, snapshot_id: int = 0) -> Any:
        if model_id in self.model_handles:
            (model_handle, _) = self.model_handles[model_id]
            self.model_handles[model_id] = (model_handle, time.time())
            return model_handle
        source_path = self._hdfs_path_of_model(model_id, snapshot_id)
        if source_path is None:
            raise Exception(
                "Could not load model for fblearner model ID: {}".format(model_id)
            )
        local_base_path = tempfile.mkdtemp()
        local_path = os.path.join(local_base_path, os.path.basename(source_path))
        self._download_model(source_path, local_path)
        config, model, data_handler = load(local_path)
        model_handle = Registry.get(config.model.getType()).create_predictor(
            config, model, data_handler
        )
        self.logger.info(
            "Predictor at path [{}] successfully loaded.".format(local_path)
        )
        self.model_handles[model_id] = (model_handle, time.time())
        return model_handle
