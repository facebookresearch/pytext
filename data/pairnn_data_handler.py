#!/usr/bin/env python3
from .data_handler import DataHandler
from torchtext import data as textdata


# TODO I don't think the model works currently, have to fix it when needed
class PairNNDataHandler(DataHandler):
    def __init__(self, *args, **kwargs) -> None:
        super(PairNNDataHandler, self).__init__(*args, **kwargs)
        assert len(self.fields) >= 3
        self.dataset = PairNNFrameDataset
        for idx, field in enumerate(
            [
                ("query_text", self.text_field),
                ("pos_text", self.text_field),
                ("neg_text", self.text_field),
            ]
        ):
            self.fields[idx] = field


class PairNNFrameDataset(textdata.Dataset):
    @staticmethod
    def sort_key(ex: textdata.Example) -> int:
        if hasattr(ex, "query_text"):
            return len(ex.query_text)
        else:
            raise ValueError("Specify the key for sorting")
