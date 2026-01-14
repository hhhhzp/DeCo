from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(
    "./InternVL3-2B",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

state_dict = torch.load(
    "dual_internvit_2b/exp_sem_ae_mlp_c128_cosine/epoch=29-step=100000.ckpt",
    map_location='cpu',
)['state_dict']
new_state_dict = {}
for key, value in state_dict.items():
    if '.lpips_loss' in key:
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
