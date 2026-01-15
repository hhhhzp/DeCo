from transformers import AutoModel
import torch

model = AutoModel.from_pretrained(
    "./InternVL3-2B",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

state_dict = torch.load(
    "dual_internvit_2b/exp_sem_layer4_r14_mlp_c32_norm_448px/epoch=16-step=60000.ckpt",
    map_location='cpu',
)['state_dict']
new_state_dict = {}
for key, value in state_dict.items():
    if '.lpips_loss' in key or '.mlp1.' in key or key.startswith('model.'):
        continue
    new_key = key
    # Remove module and _orig_mod prefixes
    if key.startswith('ema_model.'):
        new_key = key[10:]
    new_key = new_key.replace('.module.', '.')
    new_key = new_key.replace('._orig_mod.', '.')
    new_state_dict[new_key] = value
msg = model.vision_model.load_state_dict(new_state_dict)
print(msg)

model.save_pretrained("./InternVL3-2B-bak")
