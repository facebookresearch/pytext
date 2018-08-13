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

MODEL_CONFIG="$FBCODE_HOME_DIR/fblearner/flow/projects/messenger/assistant/pytext/configs/test_joint.json"

if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Using default."
elif [ $# -eq 1 ]
  then
    MODEL_CONFIG=$1
fi
echo "Using $MODEL_CONFIG as trainer config."

buck run @mode/dev-nosan pytext:trainer_main -- \
  -parameters-file "$MODEL_CONFIG"
