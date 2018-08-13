#!/bin/sh
FBCODE_HOME_DIR=""
FBCODE_DIR="/fbcode/"
FBSOURCE_DIR="/fbsource/fbcode/"

if [ -d "${HOME}${FBCODE_DIR}" ]; then
  FBCODE_HOME_DIR="${HOME}${FBCODE_DIR}"
elif [ -d "${HOME}${FBSOURCE_DIR}" ]; then
  FBCODE_HOME_DIR="${HOME}${FBSOURCE_DIR}"
else
    echo "Could not find the fbcode directory."
    exit 1
fi
buck run @mode/dev-nosan pytext:trainer_main -- \
  -parameters-file "$FBCODE_HOME_DIR/fblearner/flow/projects/messenger/assistant/pytext/configs/ensembles/test_ensemble_doc.json"
