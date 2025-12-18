#!/bin/bash

# Convert trained VAE checkpoint back to InternVL pretrained model format
# Usage: bash scripts/convert_vae_to_internvl.sh

# Configuration
VAE_CHECKPOINT="vae_stage2/exp_VAE_Encoder_Training/epoch=34-step=122000.ckpt"
ORIGINAL_MODEL="/apdcephfs/share_300000800/datamultimodal/models/InternVL3-2B"  # or path to local model
OUTPUT_PATH="vae_stage2/exp_VAE_Encoder_Training/converted_internvl_model"
rsync -av --exclude='*.safetensors' $ORIGINAL_MODEL/ $OUTPUT_PATH/
# Run conversion
python scripts/convert_vae_to_internvl.py \
    --vae_checkpoint ${VAE_CHECKPOINT} \
    --original_model ${ORIGINAL_MODEL} \
    --output_path ${OUTPUT_PATH}
cp src/models/transformer/configuration_intern_vit.py ${OUTPUT_PATH}/configuration_intern_vit.py
cp src/models/transformer/modeling_intern_vit.py ${OUTPUT_PATH}/modeling_intern_vit.py
echo "Conversion completed! Model saved to: ${OUTPUT_PATH}"
