from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(
    "./InternVL3-2B",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

s = torch.load(
    "dual_internvit_2b/exp_sem_ae_mlp_c128_cosine/epoch=29-step=100000.ckpt",
    map_location='cpu',
)
msg = model.vision_model.load_state_dict(s['state_dict'])
print(msg)
