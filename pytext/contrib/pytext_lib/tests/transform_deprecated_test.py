#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import unittest

from pytext.contrib.pytext_lib.transforms.transforms_deprecated import (
    SlotLabelTransform,
)
from pytext.utils.data import Slot


class TestSlotLabel(unittest.TestCase):
    def slot_label_test(self):
        slot_list = ["action", "time", "different"]
        transform = SlotLabelTransform(slot_list)
        slot1 = Slot("action", 7, 12)
        slot2 = Slot("time", 17, 21)
        slot3 = Slot("action", 0, 9)
        slot4 = Slot("different", 21, 31)
        result1 = transform.forward("set an alarm for 9 am", [slot1, slot2])
        result2 = transform.forward("same word different label word", [slot3, slot4])
        result3 = transform.forward("", [])
        assert result1 == [0, 0, 1, 0, 2, 2]
        assert result2 == [1, 1, 0, 3, 3]
        assert result3 == []
