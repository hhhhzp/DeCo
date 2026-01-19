from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(
    "./InternVL3-2B_sem",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

state_dict = torch.load(
    "dual_internvit_2b/exp_sem_layer4_r14_mlp_c32_c256_norm/epoch=2-step=10000.ckpt",
    map_location='cpu',
)['state_dict']
new_state_dict = {}
for key, value in state_dict.items():
    if '.lpips_loss' in key or key.startswith('ema_model.'):
        continue
    new_key = key
    # Remove module and _orig_mod prefixes
    if key.startswith('model.'):
        new_key = key[6:]
    new_key = new_key.replace('.module.', '.')
    new_key = new_key.replace('._orig_mod.', '.')
    new_state_dict[new_key] = value
msg = model.vision_model.load_state_dict(new_state_dict)
print(msg)

model.save_pretrained("exp_sem_layer4_r14_mlp_c32_c256_norm/InternVL3-2B-step10000")

# Evaluate semantic reconstruction quality
print("\nEvaluating semantic reconstruction quality...")
from PIL import Image
import torchvision.transforms as transforms

# Load and preprocess the image
image = Image.open("examples/image1.jpg").convert("RGB")
transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
image_tensor = transform(image).unsqueeze(0).to(torch.bfloat16)

with torch.no_grad():
    sem_tokens, distill_loss = model.vision_model(
        image_tensor,
        mode='semantic',
        normalize_type='imagenet',
        return_distill_loss=True,
    )
print(f"Distill Loss: {distill_loss.item():.6f}")
