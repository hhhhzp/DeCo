#!/usr/bin/env python
"""
Convert trained VAE checkpoint back to InternVL pretrained model format.

This script extracts vision_model and mlp1 from a trained VAE checkpoint
and merges them back into an InternVL pretrained model, then saves the result.

The script also:
- Copies the tokenizer from the original model
- Registers the model with AutoModel for easy loading
- Copies necessary model code files for trust_remote_code support

Usage:
    python scripts/convert_vae_to_internvl.py \
        --vae_checkpoint path/to/trained_vae.ckpt \
        --original_model path/to/original_internvl_model \
        --output_path path/to/output_internvl_model
        
After conversion, you can load the model using:
    from transformers import AutoModel, AutoTokenizer
    
    model = AutoModel.from_pretrained("path/to/output_internvl_model", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("path/to/output_internvl_model", trust_remote_code=True)
"""

import argparse
import os
import sys
import shutil
import torch
from collections import OrderedDict
from transformers import AutoTokenizer

# Add project root to path to import custom models
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.transformer.modeling_internvl_chat import InternVLChatModel
from src.models.transformer.configuration_internvl_chat import InternVLChatConfig


def extract_vae_weights(checkpoint_path):
    """
    Extract vision_model and mlp1 weights from VAE checkpoint.

    Args:
        checkpoint_path: Path to trained VAE checkpoint (.ckpt file)

    Returns:
        vision_model_state_dict: State dict for vision_model
        mlp1_state_dict: State dict for mlp1
    """
    print(f"Loading VAE checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # Extract vision_model and mlp1 weights
    vision_model_state_dict = OrderedDict()
    mlp1_state_dict = OrderedDict()

    for key, value in state_dict.items():
        # Clean up prefixes (DDP, torch.compile, etc.)
        clean_key = key
        clean_key = clean_key.replace('.module._orig_mod.', '.')
        clean_key = clean_key.replace('.module.', '.')
        clean_key = clean_key.replace('._orig_mod.', '.')

        # Extract vision_model weights
        if clean_key.startswith('vae_model.vision_model.'):
            new_key = clean_key.replace('vae_model.vision_model.', '')
            vision_model_state_dict[new_key] = value

        # Extract mlp1 weights
        elif clean_key.startswith('vae_model.mlp1.'):
            new_key = clean_key.replace('vae_model.mlp1.', '')
            mlp1_state_dict[new_key] = value

    print(f"Extracted {len(vision_model_state_dict)} vision_model parameters")
    print(f"Extracted {len(mlp1_state_dict)} mlp1 parameters")

    return vision_model_state_dict, mlp1_state_dict


def merge_weights_to_internvl(
    original_model_path,
    vision_model_state_dict,
    mlp1_state_dict,
    output_path,
):
    """
    Merge extracted weights back into InternVL model and save.

    Args:
        original_model_path: Path to original InternVL pretrained model
        vision_model_state_dict: Extracted vision_model weights
        mlp1_state_dict: Extracted mlp1 weights
        output_path: Path to save the merged model
    """
    print(f"\nLoading original InternVL model from {original_model_path}...")

    # Load original model config using custom class
    config = InternVLChatConfig.from_pretrained(original_model_path)

    # Load model directly using custom class (no AutoModel registration needed)
    model = InternVLChatModel.from_pretrained(
        original_model_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )

    print(f"✓ Loaded model type: {type(model).__name__}")

    print("Merging trained weights into model...")

    # Load vision_model weights
    missing_keys, unexpected_keys = model.vision_model.load_state_dict(
        vision_model_state_dict, strict=False
    )

    if missing_keys:
        print(f"Warning: Missing keys in vision_model: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in vision_model: {unexpected_keys}")

    # Load mlp1 weights
    missing_keys, unexpected_keys = model.mlp1.load_state_dict(
        mlp1_state_dict, strict=True
    )

    if missing_keys:
        print(f"Warning: Missing keys in mlp1: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in mlp1: {unexpected_keys}")

    print(f"\nSaving merged model to {output_path}...")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save the merged model
    model.save_pretrained(
        output_path,
        safe_serialization=True,  # Use safetensors format
    )

    # Add auto_map to config for AutoModel registration
    config.auto_map = {
        "AutoConfig": "configuration_internvl_chat.InternVLChatConfig",
        "AutoModel": "modeling_internvl_chat.InternVLChatModel",
        "AutoModelForCausalLM": "modeling_internvl_chat.InternVLChatModel",
    }

    # Save the config with auto_map
    config.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert trained VAE checkpoint back to InternVL pretrained model format"
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        required=True,
        help="Path to trained VAE checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--original_model",
        type=str,
        required=True,
        help="Path to original InternVL pretrained model directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the converted InternVL model",
    )

    args = parser.parse_args()

    # Validate input paths
    if not os.path.exists(args.vae_checkpoint):
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_checkpoint}")

    if not os.path.exists(args.original_model):
        raise FileNotFoundError(f"Original model not found: {args.original_model}")

    print("=" * 80)
    print("VAE to InternVL Conversion Script")
    print("=" * 80)
    print(f"VAE Checkpoint: {args.vae_checkpoint}")
    print(f"Original Model: {args.original_model}")
    print(f"Output Path: {args.output_path}")
    print("=" * 80)

    # Step 1: Extract weights from VAE checkpoint
    vision_model_state_dict, mlp1_state_dict = extract_vae_weights(args.vae_checkpoint)

    # Step 2: Merge weights back into InternVL model
    merge_weights_to_internvl(
        args.original_model,
        vision_model_state_dict,
        mlp1_state_dict,
        args.output_path,
    )

    print("\n" + "=" * 80)
    print("Conversion completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
