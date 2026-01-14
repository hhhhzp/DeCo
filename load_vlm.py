from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(
    "./InternVL3-2B",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
