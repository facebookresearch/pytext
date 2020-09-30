#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

ROBERTA_BASE_TORCH = "roberta_base_torch"
ROBERTA_PUBLIC = "roberta_public"
XLMR_BASE = "xlmr_base"
XLMR_DUMMY = "xlmr_dummy"

URL = {
    ROBERTA_BASE_TORCH: "https//dl.fbaipublicfiles.com/pytext/models/roberta/roberta_base_torch.pt",  # noqa
    ROBERTA_PUBLIC: "https//dl.fbaipublicfiles.com/pytext/models/roberta/roberta_public.pt1",  # noqa
    XLMR_BASE: "https://dl.fbaipublicfiles.com/pytext/models/xlm_r/checkpoint_base_1500k.pt",  # noqa
    XLMR_DUMMY: "https://dl.fbaipublicfiles.com/pytext/models/xlm_r/xlmr_dummy.pt",  # noqa
}
