#!/usr/bin/env python3
from enum import Enum

from pytext.config import ConfigBase
from pytext.config.module_config import LSTMParams
from pytext.trainers.trainer import TrainerConfig


class CompositionalType(Enum):
    BLSTM = "blstm"
    SUM = "sum"


class AblationParams(ConfigBase):
    use_buffer: bool = True
    use_stack: bool = True
    use_action: bool = True


class RNNGConstraints(ConfigBase):
    intent_slot_nesting: bool = True
    ignore_loss_for_unsupported: bool = False
    no_slots_inside_unsupported: bool = True
    # Path to ontology JSON
    ontology: str = ""


class RNNGConfig(ConfigBase):
    lstm: LSTMParams = LSTMParams()
    ablation: AblationParams = AblationParams()
    constraints: RNNGConstraints = RNNGConstraints()
    max_train_num: int = -1
    max_dev_num: int = -1
    max_test_num: int = -1
    max_open_NT: int = 10
    dropout: float = 0.1
    compositional_type: CompositionalType = CompositionalType.BLSTM
    all_metrics: bool = False
    use_cpp: bool = False


# TODO only a placeholder now
class Seq2SeqConfig(ConfigBase):
    pass


# TODO move it to generic place when refactoring rnng
class CompositionalTrainerConfig(TrainerConfig, ConfigBase):
    # num of workers for hogwild training
    num_workers: int = 1
