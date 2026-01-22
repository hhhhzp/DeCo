from transformers import AutoModel
import torch
import shutil
import os

# ============ Configuration / Naming Rules ============
# Model paths
PRETRAINED_MODEL_PATH = "./InternVL3-2B_sem_new"
CHECKPOINT_PATH = (
    "dual_internvit_2b/exp_sem_gen_gate_c256_new_stage2_448px/epoch=0-step=35000.ckpt"
)

# Output paths
OUTPUT_BASE_DIR = "exp_sem_gen_gate_c256_new_stage2_448px"
OUTPUT_MODEL_NAME = "InternVL3-2B-step35000-model"
OUTPUT_EMA_NAME = "InternVL3-2B-step35000-emamodel"
OUTPUT_MODEL_PATH = f"{OUTPUT_BASE_DIR}/{OUTPUT_MODEL_NAME}"
OUTPUT_EMA_PATH = f"{OUTPUT_BASE_DIR}/{OUTPUT_EMA_NAME}"

# Test image path
TEST_IMAGE_PATH = "examples/image1.jpg"

# Model configuration
MODEL_DTYPE = torch.bfloat16
IMAGE_SIZE = (448, 448)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# State dict key prefixes
MODEL_PREFIX = 'model.'
EMA_MODEL_PREFIX = 'ema_model.'
SKIP_KEY_PATTERNS = ('.lpips_loss', '.teacher_mlp')

# Files to copy from pretrained model (excluding safetensors)
FILES_TO_COPY = [
    'added_tokens.json',
    'merges.txt',
    'preprocessor_config.json',
    'special_tokens_map.json',
    'tokenizer_config.json',
    'tokenizer.json',
    'vocab.json',
]
# ======================================================

model = AutoModel.from_pretrained(
    PRETRAINED_MODEL_PATH,
    dtype=MODEL_DTYPE,
    trust_remote_code=True,
)

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location='cpu',
)['state_dict']

# Process state dict in one pass - separate into vision_model and mlp1 for both model and ema_model
vision_model_dict = {}
mlp1_dict = {}
vision_ema_dict = {}
mlp1_ema_dict = {}

for key, value in state_dict.items():
    if any(pattern in key for pattern in SKIP_KEY_PATTERNS):
        continue

    # Determine if this is EMA or regular model
    is_ema = key.startswith(EMA_MODEL_PREFIX)

    # Remove prefix
    new_key = key
    if is_ema:
        new_key = key[len(EMA_MODEL_PREFIX) :]
    elif key.startswith(MODEL_PREFIX):
        new_key = key[len(MODEL_PREFIX) :]

    # Clean up module and _orig_mod prefixes
    new_key = new_key.replace('.module.', '.')
    new_key = new_key.replace('._orig_mod.', '.')

    if is_ema:
        vision_ema_dict[new_key] = value
    else:
        vision_model_dict[new_key] = value


# Helper function to copy additional files
def copy_additional_files(src_dir, dst_dir, files_list):
    """Copy additional files from source to destination directory"""
    os.makedirs(dst_dir, exist_ok=True)
    copied_files = []
    missing_files = []

    for filename in files_list:
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)

        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            copied_files.append(filename)
        else:
            missing_files.append(filename)

    return copied_files, missing_files


# Helper function to load and save model
def load_and_save_model(model, vision_dict, mlp1_dict, output_path, model_type="Model"):
    """Load vision_model and mlp1, then save to output_path"""
    print(mlp1_dict)
    print(f"\n{'='*50}")
    print(f"Loading {model_type}...")
    print(f"{'='*50}")

    # Load vision_model
    msg_vision = model.vision_model.load_state_dict(vision_dict)
    print(f"{model_type} vision_model load result:", msg_vision)

    msg_mlp1 = model.mlp1.load_state_dict(model.vision_model.mlp1.state_dict())
    print(f"{model_type} mlp1 load result:", msg_mlp1)

    # Save model
    model.save_pretrained(output_path)
    print(f"\n{model_type} saved to: {output_path}")

    # Copy additional files
    print(f"Copying additional files to {output_path}...")
    copied, missing = copy_additional_files(
        PRETRAINED_MODEL_PATH, output_path, FILES_TO_COPY
    )
    print(f"Copied {len(copied)} files: {', '.join(copied)}")
    if missing:
        print(f"Missing {len(missing)} files: {', '.join(missing)}")


# Save model version
load_and_save_model(model, vision_model_dict, mlp1_dict, OUTPUT_MODEL_PATH, "Model")

# Save ema_model version
load_and_save_model(model, vision_ema_dict, mlp1_ema_dict, OUTPUT_EMA_PATH, "EMA Model")

# Evaluate semantic reconstruction quality
print("\nEvaluating semantic reconstruction quality...")
from PIL import Image
import torchvision.transforms as transforms

# Load and preprocess the image
image = Image.open(TEST_IMAGE_PATH).convert("RGB")
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ]
)
image_tensor = transform(image).unsqueeze(0).to(MODEL_DTYPE)

with torch.no_grad():
    sem_tokens, distill_loss = model.vision_model(
        image_tensor,
        mode='semantic',
        normalize_type='imagenet',
        return_distill_loss=True,
    )
print(f"Distill Loss: {distill_loss.item():.6f}")
