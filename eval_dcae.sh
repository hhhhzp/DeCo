#!/bin/bash

# DCAE Decoder Evaluation Script
# This script evaluates the DCAE decoder on ImageNet validation set

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration file
CONFIG_FILE="configs_flow/dcae_decoder_eval.yaml"

# Optional: Checkpoint path for trained decoder
# CHECKPOINT_PATH="path/to/your/checkpoint.ckpt"

# Run evaluation
python eval_dcae.py predict \
    --config ${CONFIG_FILE}
# Alternative: Run validation (with metrics computation)
# python eval_dcae.py validate \
#     --config ${CONFIG_FILE}
