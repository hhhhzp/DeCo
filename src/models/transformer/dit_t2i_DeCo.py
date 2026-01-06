import torch
import torch.nn as nn
import warnings

from functools import lru_cache
from src.models.layers.attention_op import attention
from src.models.layers.rope import (
    apply_rotary_emb,
    precompute_freqs_cis_ex2d as precompute_freqs_cis_2d,
)
from src.models.layers.time_embed import TimestepEmbedder as TimestepEmbedder
from src.models.layers.patch_embed import Embed as Embed
from src.models.layers.swiglu import SwiGLU as FeedForward

# from src.models.layers.rmsnorm import RMSNorm as Norm
from src.models.transformer.configuration_internvl_chat import InternVLChatConfig
from src.models.transformer.modeling_intern_vit import InternVisionModel
from src.models.transformer.configuration_intern_vit import InternVisionConfig
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
# import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, pos) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q.contiguous())
        k = self.k_norm(k.contiguous())
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)

        q = q.view(B, self.num_heads, -1, C // self.num_heads)  # B, H, N, Hc
        k = k.view(
            B, self.num_heads, -1, C // self.num_heads
        ).contiguous()  # B, H, N, Hc
        v = v.view(B, self.num_heads, -1, C // self.num_heads).contiguous()

        x = attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlattenDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        groups,
        mlp_ratio=4,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, pos):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(x).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class NerfEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            nn.Linear(in_channels + max_freqs**2, hidden_size_input, bias=True),
        )

    @lru_cache
    def fetch_pos(self, patch_size, device, dtype):
        pos = precompute_freqs_cis_2d(self.max_freqs**2 * 2, patch_size, patch_size)
        pos = pos[None, :, :].to(device=device, dtype=dtype)
        return pos

    def forward(self, inputs):
        B, P2, C = inputs.shape
        patch_size = int(P2**0.5)
        device = inputs.device
        dtype = inputs.dtype
        dct = self.fetch_pos(patch_size, device, dtype)
        dct = dct.repeat(B, 1, 1)
        inputs = torch.cat([inputs, dct], dim=-1)
        inputs = self.embedder(inputs)
        return inputs


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size, expansion_ratio=4):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expansion_ratio),
            nn.SiLU(),
            nn.Linear(hidden_size * expansion_ratio, hidden_size),
        )

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class LatentConnectorModule(nn.Module):
    def __init__(self, hidden_size, out_channels, expansion_ratio=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.mlp_blocks = nn.Sequential(
            ResidualMLPBlock(hidden_size, expansion_ratio=expansion_ratio),
            ResidualMLPBlock(hidden_size, expansion_ratio=expansion_ratio),
            ResidualMLPBlock(hidden_size, expansion_ratio=expansion_ratio),
        )
        if hidden_size == out_channels:
            self.final_proj = nn.Identity()
        else:
            self.final_proj = nn.Linear(hidden_size, out_channels)

        # 执行初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        for block in self.mlp_blocks:
            if hasattr(block, 'mlp') and isinstance(block.mlp[-1], nn.Linear):
                nn.init.constant_(block.mlp[-1].weight, 0)
                if block.mlp[-1].bias is not None:
                    nn.init.constant_(block.mlp[-1].bias, 0)

    def forward(self, x):
        # x: [B, N, hidden_size]
        x = self.mlp_blocks(x)
        x = self.final_proj(x)
        return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        patch_size,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.patch_size = patch_size

        self.cond_embed = nn.Linear(z_channels, patch_size**2 * model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(
                ResBlock(
                    model_channels,
                )
            )

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        c = self.cond_embed(c)

        y = c.reshape(c.shape[0], self.patch_size**2, -1)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x)


class PixelDecoder(nn.Module):
    """Pixel Decoder for diffusion model."""

    def __init__(
        self,
        in_channels=4,
        latent_channel=64,
        hidden_size=1152,
        hidden_size_x=64,
        num_groups=12,
        num_encoder_blocks=18,
        num_decoder_blocks=4,
        patch_size=14,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channel = latent_channel
        self.hidden_size = hidden_size
        self.hidden_size_x = hidden_size_x
        self.num_groups = num_groups
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks
        self.patch_size = patch_size

        # Learnable tokens for DiT (replacing text condition)
        num_learnable_tokens = 16  # Can be adjusted
        self.learnable_tokens = nn.Parameter(
            torch.zeros(1, num_learnable_tokens, hidden_size), requires_grad=True
        )
        # Use smaller initialization to prevent instability
        nn.init.normal_(self.learnable_tokens, mean=0.0, std=0.02)

        # Embedders
        self.s_embedder = Embed(self.latent_channel, hidden_size, bias=True)
        self.x_embedder = NerfEmbedder(in_channels, hidden_size_x, max_freqs=8)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        self.blocks = nn.ModuleList(
            [
                FlattenDiTBlock(self.hidden_size, self.num_groups)
                for _ in range(self.num_encoder_blocks)
            ]
        )

        # Decoder network with time embedding
        self.dec_net = SimpleMLPAdaLN(
            in_channels=hidden_size_x,
            model_channels=hidden_size_x,
            out_channels=self.in_channels,  # for vlb loss
            z_channels=self.hidden_size,
            num_res_blocks=num_decoder_blocks,
            patch_size=patch_size,
            grad_checkpointing=False,
        )

        self.precompute_pos = dict()
        self.initialize_weights()

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(
                self.hidden_size // self.num_groups, height, width
            ).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward_condition(self, latent, device):
        """
        Process latent features through DiT blocks.
        :param latent: latent features [B, N, latent_channel]
        :param device: device for computation
        :return: processed features [B, N, hidden_size]
        """
        B = latent.shape[0]
        grid_size = int(latent.shape[1] ** 0.5)
        xpos = self.fetch_pos(grid_size, grid_size, device)

        # Expand learnable tokens for batch
        learnable_tokens = self.learnable_tokens.expand(B, -1, -1)

        # Embed latent features
        s = self.s_embedder(latent)

        # Concatenate learnable tokens (like cls token) with spatial tokens
        s = torch.cat([learnable_tokens, s], dim=1)

        # Create position encoding: center position for learnable tokens, rotary for spatial tokens
        num_learnable = learnable_tokens.shape[1]
        # xpos shape: [grid_size*grid_size, hidden_size//num_groups]
        # Use center position encoding for learnable tokens
        center_idx = grid_size * grid_size // 2  # Middle position in the spatial grid
        center_pos = xpos[center_idx : center_idx + 1].expand(num_learnable, -1)
        full_pos = torch.cat([center_pos, xpos], dim=0)

        # DiT Block iteration
        for block in self.blocks:
            s = block(s, full_pos)

        # Remove learnable tokens and return only spatial tokens
        s = s[:, num_learnable:, :]

        return s

    def forward(self, x, t, s):
        """
        Decode pixels from latent features.
        :param x: noisy input [B, C, H, W]
        :param t: timestep [B]
        :param s: condition features [B, N, hidden_size]
        :return: predicted output [B, C, H, W]
        """
        B, _, H, W = x.shape
        x = torch.nn.functional.unfold(
            x, kernel_size=self.patch_size, stride=self.patch_size
        ).transpose(1, 2)
        t = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)

        s = torch.nn.functional.silu(t + s)
        batch_size, length, _ = s.shape
        x = x.reshape(batch_size * length, self.in_channels, self.patch_size**2)
        x = x.transpose(1, 2)
        s = s.view(batch_size * length, self.hidden_size)
        x = self.x_embedder(x)

        x = self.dec_net(x, s)

        x = x.transpose(1, 2)
        x = x.reshape(batch_size, length, -1)
        x = torch.nn.functional.fold(
            x.transpose(1, 2).contiguous(),
            (H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return x


class PixNerDiT(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_groups=12,
        hidden_size=1152,
        hidden_size_x=64,
        num_encoder_blocks=18,
        num_decoder_blocks=4,
        patch_size=2,
        weight_path=None,
        load_ema=False,
        config_path=None,
        select_layer=-1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.decoder_hidden_size = hidden_size_x
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks
        self.num_blocks = self.num_encoder_blocks + self.num_decoder_blocks
        self.select_layer = select_layer

        # Load config
        config = InternVLChatConfig.from_pretrained(config_path)
        vision_config = config.vision_config
        vision_config.drop_path_rate = 0.0
        vision_config.num_hidden_layers = select_layer
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.downsample_ratio = 0.5
        self.latent_channel = 32
        self.patch_size = vision_config.patch_size

        # ============================================================
        # Vision Encoder
        # ============================================================
        self.vision_model = InternVisionModel(vision_config)

        # ============================================================
        # Connector
        # ============================================================
        # First connector: vision features -> llm hidden size
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                llm_hidden_size,
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        # Second connector: llm hidden size -> latent space
        self.latent_projector = nn.Sequential(
            nn.LayerNorm(vit_hidden_size),
            nn.Linear(vit_hidden_size, vit_hidden_size),
            nn.GELU(),
            nn.Linear(vit_hidden_size, self.latent_channel),
        )

        # ============================================================
        # Pixel Decoder
        # ============================================================
        self.pixel_decoder = PixelDecoder(
            in_channels=in_channels,
            latent_channel=self.latent_channel,
            hidden_size=hidden_size,
            hidden_size_x=hidden_size_x,
            num_groups=num_groups,
            num_encoder_blocks=num_encoder_blocks,
            num_decoder_blocks=num_decoder_blocks,
            patch_size=self.patch_size,
        )

        self.weight_path = weight_path
        self.load_ema = load_ema

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_vision_feature(self, pixel_values):
        """
        Extract features from vision model with pixel shuffle downsampling.
        :param pixel_values: input image [B, C, H, W]
        :return: vit_embeds [B, num_patches, decoder_hidden_size]
        """
        pixel_values = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(
            pixel_values * 0.5 + 0.5
        )
        vit_embeds = self.vision_model(
            pixel_values=pixel_values, output_hidden_states=False, return_dict=True
        ).last_hidden_state
        vit_embeds = vit_embeds[:, 1:, :]  # Remove CLS token
        return vit_embeds

    def extract_feature(self, pixel_values):
        """
        Extract features from vision model with pixel shuffle downsampling.
        :param pixel_values: input image [B, C, H, W]
        :return: vit_embeds [B, num_patches, decoder_hidden_size]
        """
        pixel_values = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(
            pixel_values * 0.5 + 0.5
        )
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]  # Remove CLS token

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def forward_condition(self, x, vit_embeds=None):
        """
        Extract and process condition features.
        :param x: input image [B, C, H, W]
        :param vit_embeds: pre-extracted vision features (optional)
        :return: condition features [B, N, hidden_size]
        """
        # Extract features and project to latent space [B, N, latent_channel]
        if vit_embeds is None:
            latent = self.latent_projector(self.extract_vision_feature(x))
        else:
            latent = self.latent_projector(vit_embeds)

        # Process through pixel decoder
        s = self.pixel_decoder.forward_condition(latent, x.device)
        return s

    def forward(self, x, t, s):
        """
        Forward pass through the complete model.
        :param x: noisy input [B, C, H, W]
        :param t: timestep [B]
        :param s: condition features [B, N, hidden_size]
        :return: predicted output [B, C, H, W]
        """
        return self.pixel_decoder(x, t, s)
