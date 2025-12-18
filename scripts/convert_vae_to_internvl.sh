#!/bin/bash

# Convert trained VAE checkpoint back to InternVL pretrained model format
# Usage: bash scripts/convert_vae_to_internvl.sh

# Configuration
VAE_CHECKPOINT="path/to/your/trained_vae.ckpt"
ORIGINAL_MODEL="OpenGVLab/InternVL2-8B"  # or path to local model
OUTPUT_PATH="output/converted_internvl_model"

# Run conversion
python scripts/convert_vae_to_internvl.py \
    --vae_checkpoint ${VAE_CHECKPOINT} \
    --original_model ${ORIGINAL_MODEL} \
    --output_path ${OUTPUT_PATH}

echo "Conversion completed! Model saved to: ${OUTPUT_PATH}"
