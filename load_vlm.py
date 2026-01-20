from transformers import AutoModel
import torch
import shutil
import os

# ============ Configuration / Naming Rules ============
# Model paths
PRETRAINED_MODEL_PATH = "./InternVL3-2B_sem"
CHECKPOINT_PATH = "dual_internvit_2b/exp_sem_layer4_r14_mlp_c32_c256_norm_448px/epoch=28-step=100000.ckpt"

# Output paths
OUTPUT_BASE_DIR = "exp_sem_layer4_r14_mlp_c32_c256_norm"
OUTPUT_MODEL_NAME = "InternVL3-2B-step100000-model"
OUTPUT_EMA_NAME = "InternVL3-2B-step100000-emamodel"
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
SKIP_KEY_PATTERN = '.lpips_loss'

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
# Process model version
new_state_dict_model = {}
for key, value in state_dict.items():
    if SKIP_KEY_PATTERN in key or key.startswith(EMA_MODEL_PREFIX):
        continue
    new_key = key
    # Remove module and _orig_mod prefixes
    if key.startswith(MODEL_PREFIX):
        new_key = key[len(MODEL_PREFIX) :]
    new_key = new_key.replace('.module.', '.')
    new_key = new_key.replace('._orig_mod.', '.')
    new_state_dict_model[new_key] = value

# Process ema_model version
new_state_dict_ema = {}
for key, value in state_dict.items():
    if SKIP_KEY_PATTERN in key or not key.startswith(EMA_MODEL_PREFIX):
        continue
    new_key = key
    # Remove ema_model prefix
    if key.startswith(EMA_MODEL_PREFIX):
        new_key = key[len(EMA_MODEL_PREFIX) :]
    new_key = new_key.replace('.module.', '.')
    new_key = new_key.replace('._orig_mod.', '.')
    new_state_dict_ema[new_key] = value


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


# Save model version
msg = model.vision_model.load_state_dict(new_state_dict_model)
print("Model version:", msg)
model.save_pretrained(OUTPUT_MODEL_PATH)
print(f"\nCopying additional files to {OUTPUT_MODEL_PATH}...")
copied, missing = copy_additional_files(
    PRETRAINED_MODEL_PATH, OUTPUT_MODEL_PATH, FILES_TO_COPY
)
print(f"Copied {len(copied)} files: {', '.join(copied)}")
if missing:
    print(f"Missing {len(missing)} files: {', '.join(missing)}")

# Save ema_model version
msg_ema = model.vision_model.load_state_dict(new_state_dict_ema)
print("\nEMA model version:", msg_ema)
model.save_pretrained(OUTPUT_EMA_PATH)
print(f"\nCopying additional files to {OUTPUT_EMA_PATH}...")
copied_ema, missing_ema = copy_additional_files(
    PRETRAINED_MODEL_PATH, OUTPUT_EMA_PATH, FILES_TO_COPY
)
print(f"Copied {len(copied_ema)} files: {', '.join(copied_ema)}")
if missing_ema:
    print(f"Missing {len(missing_ema)} files: {', '.join(missing_ema)}")

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
