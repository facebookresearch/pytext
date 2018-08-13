#!/bin/sh
CUDA_LAUNCH_BLOCKING=1 buck run  @mode/dev-nosan  pytext/rnng:train_parser -- \
-parameters-file "$HOME/fbsource/fbcode/fblearner/flow/projects/messenger/assistant/pytext/configs/test_rnng.json"
