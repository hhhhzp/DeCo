from transformers import AutoModel
import torch

model = (
    AutoModel.from_pretrained(
        "exp_sem_layer4_r14_mlp_c32_c256_norm/InternVL3-2B-step95000",
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
        transforms.Resize((224, 224)),
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

# distill_loss shape: [1, N, C], dtype: bfloat16
# Convert to float32 for better precision in calculations
distill_loss = distill_loss.float()

# Average over C dimension to get [1, N]
loss_per_position = distill_loss.mean(dim=-1).squeeze(0).cpu().numpy()  # [N]

print(f"Overall Distill Loss: {distill_loss.mean().item():.6f}")
print(f"Loss shape after C-dim average: {loss_per_position.shape}")
print(f"Loss range: [{loss_per_position.min():.6f}, {loss_per_position.max():.6f}]")


# Plot ASCII curve in terminal
def plot_ascii_curve(values, height=20, width=60):
    """Plot a simple ASCII curve in terminal"""
    import numpy as np

    # Normalize values to fit in the plot height
    min_val, max_val = values.min(), values.max()
    if max_val - min_val < 1e-8:
        print("All values are the same, cannot plot curve.")
        return

    # Resample to fit width if needed
    if len(values) > width:
        indices = np.linspace(0, len(values) - 1, width).astype(int)
        values = values[indices]

    normalized = (values - min_val) / (max_val - min_val)
    scaled = (normalized * (height - 1)).astype(int)

    # Create plot grid
    grid = [[' ' for _ in range(len(values))] for _ in range(height)]

    # Fill in the curve
    for i, val in enumerate(scaled):
        grid[height - 1 - val][i] = '●'

    # Print the plot
    print(f"\nLoss Curve (N={len(loss_per_position)} positions):")
    print(f"Max: {max_val:.6f} ┐")
    for row in grid:
        print("            │" + ''.join(row))
    print(f"Min: {min_val:.6f} └" + "─" * len(values))
    print(f"            Position: 0 → {len(loss_per_position)-1}")


plot_ascii_curve(loss_per_position)
