from transformers import AutoModel
import torch

model = (
    AutoModel.from_pretrained(
        "dual_internvit_2b/exp_sem_layer4_r14_mlp_c32_c256_norm_448px/epoch=26-step=95000.ckpt",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    .cuda()
    .eval()
)
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
image_tensor = transform(image).cuda().unsqueeze(0).to(torch.bfloat16)

with torch.no_grad():
    sem_tokens, distill_loss = model.vision_model(
        image_tensor,
        mode='semantic',
        normalize_type='imagenet',
        return_distill_loss=True,
    )
print(f"Distill Loss: {distill_loss.item():.6f}")
