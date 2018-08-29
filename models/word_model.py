#!/usr/bin/env python3

from typing import Union

from pytext.common.constants import DatasetFieldName
from pytext.config.field_config import FeatureConfig
from pytext.config import ConfigBase
from pytext.data import CommonMetadata
from pytext.models.crf import CRF
from pytext.models.model import Model
from pytext.models.projections.mlp_projection import MLPProjection
from pytext.models.representations.bilstm_slot_attn import BiLSTMSlotAttention
from pytext.models.representations.biseqcnn import BSeqCNNRepresentation


class WordTaggingModel(Model):
    """
    Word tagging model.
    """
    class Config(ConfigBase):
        representation: Union[BiLSTMSlotAttention.Config, BSeqCNNRepresentation.Config]
        proj_config: MLPProjection.Config = MLPProjection.Config()
        use_crf: bool = False

    @classmethod
    def from_config(
        cls,
        model_config: Config,
        feat_config: FeatureConfig,
        metadata: CommonMetadata,
    ):
        model = super().from_config(model_config, feat_config, metadata)
        word_label_num = metadata.labels[DatasetFieldName.WORD_LABEL_FIELD].vocab_size
        model.crf = CRF(word_label_num) if model_config.use_crf else None
        return model
