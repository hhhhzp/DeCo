# --------------------------------------------------------
# Copyright (c) 2025, OpenGVLab. All rights reserved.
# Licensed under The MIT License [see LICENSE for details]
# UniFlow-(InternViT)
# --------------------------------------------------------

import ast
import math
import os
from collections import OrderedDict
from functools import lru_cache
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import Block
from torchvision.transforms import Normalize
from src.models.layers.attention_op import attention
from src.models.layers.rope import (
    apply_rotary_emb,
    precompute_freqs_cis_ex2d as precompute_freqs_cis_2d,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from .configuration_uniflow import UniFlowVisionConfig

try:
    from src.models.uniflow.flash_attention import FlashAttention

    has_flash_attn = True
except Exception as e:
    print(e)
    print('FlashAttention is not installed.')
    has_flash_attn = False

try:
    from apex.normalization import FusedRMSNorm

    UniFlowRMSNorm = FusedRMSNorm  # noqa

    logger.info(
        'Discovered apex.normalization.FusedRMSNorm - will use it instead of UniFlowRMSNorm'
    )
except ImportError:
    # using the normal UniFlowRMSNorm
    pass
except Exception:
    logger.warning(
        'discovered apex but it failed to load, falling back to UniFlowRMSNorm'
    )
    pass

logger = logging.get_logger(__name__)

import warnings

warnings.filterwarnings("ignore")


#############################################################
#                 UniFlow Modules
#############################################################


def p2l_transform_tensor(x, patch_size):
    """
    Transform from pixel space to latent space
    [B, C, H, W] -> [B, * H//patch_size * W//patch_size, C*patch_size*patch_size]
    """
    B, C, H, W = x.shape
    return rearrange(
        x,
        "b c (h1 h2) (w1 w2) -> b (h1 w1) (c h2 w2)",
        h1=H // patch_size,
        h2=patch_size,
        w1=W // patch_size,
        w2=patch_size,
    )


def l2p_transform_tensor(x, patch_size, img_size=None):
    """
    Transform from latent space to pixel space
    [B, H//patch_size * W//patch_size, C*tubelet_size*patch_size*patch_size] -> [B, C, H, W]
    """
    B = x.shape[0]
    num_patches = x.shape[1]
    C = x.shape[2] // (patch_size * patch_size)

    # Auto-infer img_size from num_patches (assuming square image)
    if img_size is None:
        grid_size = int(num_patches**0.5)
        img_size = grid_size * patch_size

    return rearrange(
        x,
        "b (h1 w1) (c h2 w2) -> b c (h1 h2) (w1 w2)",
        h1=img_size // patch_size,
        h2=patch_size,
        w1=img_size // patch_size,
        w2=patch_size,
        c=C,
    )


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class UniFlowRMSNorm(nn.Module):
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


NORM2FN = {
    'rms_norm': UniFlowRMSNorm,
    'layer_norm': nn.LayerNorm,
}


class Attention(nn.Module):
    """Attention module for FlattenDiTBlock"""

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

        self.q_norm = UniFlowRMSNorm(self.head_dim)
        self.k_norm = UniFlowRMSNorm(self.head_dim)
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


class FeedForward(nn.Module):
    """FeedForward module for FlattenDiTBlock"""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(torch.nn.functional.silu(x1) * x2)


class FlattenDiTBlock(nn.Module):
    """FlattenDiT Block with RMSNorm, Attention and FeedForward"""

    def __init__(
        self,
        hidden_size,
        groups,
        mlp_ratio=4,
    ):
        super().__init__()
        self.norm1 = UniFlowRMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = UniFlowRMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)

    def forward(self, x, pos):
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.mlp(self.norm2(x))
        return x


class UniFlowVisionEmbeddings(nn.Module):
    def __init__(self, config: UniFlowVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_positions, self.embed_dim)
        )

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = (
            pos_embed.float()
            .reshape(
                1,
                self.image_size // self.patch_size,
                self.image_size // self.patch_size,
                -1,
            )
            .permute(0, 3, 1, 2)
        )
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
            .to(target_dtype)
        )
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values
        )  # shape = [batch*temporal, channel, width, height]  [batch*temporal, channel*patch*patch, width//patch, height//patch]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(
            1, 2
        )  # [batch, seq_le=1024, dim]
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat(
            [
                self.position_embedding[:, :1, :],
                self._get_pos_embed(self.position_embedding[:, 1:, :], height, width),
            ],
            dim=1,
        )
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class UniFlowAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: UniFlowVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = UniFlowRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = UniFlowRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        if self.use_flash_attn:
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _naive_attn(
        self,
        x,
        attn_mask=None,
    ):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = (
                self.q_norm(q.transpose(1, 2).flatten(-2, -1))
                .view(B_, N_, H_, D_)
                .transpose(1, 2)
            )
            k = (
                self.k_norm(k.transpose(1, 2).flatten(-2, -1))
                .view(B_, N_, H_, D_)
                .transpose(1, 2)
            )

        attn_bias = torch.zeros(N, N, dtype=q.dtype, device=q.device)
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn += attn_bias  # masking
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(
        self,
        x,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
    ):
        qkv = self.qkv(x)
        qkv = rearrange(
            qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads
        )

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            causal=False,
        )
        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)
        return outs

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask=None,
    ) -> torch.Tensor:
        x = (
            self._naive_attn(hidden_states, attn_mask=attn_mask)
            if not self.use_flash_attn
            else self._flash_attn(hidden_states, attn_mask=attn_mask)
        )
        return x


class UniFlowMLP(nn.Module):
    def __init__(self, config: UniFlowVisionConfig):
        super().__init__()
        self.config = config
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class UniFlowVisionEncoderLayer(nn.Module):
    def __init__(self, config: UniFlowVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = UniFlowAttention(config)
        self.mlp = UniFlowMLP(config)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask=None,
    ) -> Tuple[
        torch.FloatTensor,
        Optional[torch.FloatTensor],
        Optional[Tuple[torch.FloatTensor]],
    ]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        hidden_states = hidden_states + self.drop_path1(
            self.attn(self.norm1(hidden_states), attn_mask=attn_mask) * self.ls1
        )

        hidden_states = hidden_states + self.drop_path2(
            self.mlp(self.norm2(hidden_states)) * self.ls2
        )

        return hidden_states


class UniFlowVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`UniFlowEncoderLayer`].
    Args:
        config (`UniFlowConfig`):
            The corresponding vision configuration for the `UniFlowEncoder`.
    """

    def __init__(self, config: UniFlowVisionConfig):
        super().__init__()
        self.config = config
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)
        ]
        self.layers = nn.ModuleList(
            [
                UniFlowVisionEncoderLayer(config, dpr[idx])
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        attn_mask=None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    attn_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attn_mask,
                )
            hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )


class Distill_Adapter(nn.Module):
    def __init__(self, in_channels=1408, out_channels=3200, norm_layer=nn.LayerNorm):
        super().__init__()

        self.head = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(self.head(x))
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


class FlowDecoder(nn.Module):
    """patch-wise pixel flow decoder (rectified flow)"""

    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        grad_checkpointing=False,
        num_sampling_steps='10',
        train_schedule='fat_lognormal',
        use_cfg=False,
        noise_concat=False,
        patch_size=14,
        img_size=224,
        max_freqs=8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.target_channels = target_channels

        # configs
        self.use_cfg = use_cfg
        self.train_schedule = train_schedule
        self.num_sampling_steps = int(num_sampling_steps)
        self.noise_concat = noise_concat
        print(f"Sampling Step: {self.num_sampling_steps}")
        print(f"Train Schedule: {self.train_schedule}")

        # mlp head (latent to pixel)
        self.in_channels = (
            target_channels + z_channels if noise_concat else target_channels
        )

        # NerfEmbedder for condition tokens
        self.nerf_embedder = NerfEmbedder(
            in_channels=z_channels,
            hidden_size_input=z_channels,
            max_freqs=max_freqs,
        )

        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
        )

    def forward_train(self, x1, z):
        """
        Training forward pass for flow matching.
        Args:
            x1: target clean data [B, N, C] where C = target_channels
            z: condition from encoder [B, N, C_z]
        Returns:
            loss: flow matching loss
        """
        b, n, c = x1.shape
        assert (
            c == self.target_channels
        ), f"Expected {self.target_channels} channels, got {c}"

        # Apply NerfEmbedder to condition tokens
        z = self.nerf_embedder(z)

        # Flatten batch and sequence dimensions
        x1 = x1.reshape(b * n, c)
        z = z.reshape(b * n, -1)

        # Sample noise x0
        x0 = torch.randn_like(x1)

        # Sample timestep t using logit-normal distribution
        # Logit-Normal: t = sigmoid(nt) where nt ~ N(0, 1)
        nt = torch.randn((b * n,), device=x1.device)
        t = torch.sigmoid(nt)
        # 90% logit-normal, 10% uniform
        t = torch.where(torch.rand_like(t) <= 0.9, t, torch.rand_like(t))

        # Interpolate between x0 and x1: x_t = t * x1 + (1 - t) * x0
        t_expanded = t.view(-1, 1)
        x_t = t_expanded * x1 + (1 - t_expanded) * x0

        # Target velocity: v_target = x1 - x0
        v_target = x1 - x0

        # Predict velocity
        timesteps = t * 1000  # scale to [0, 1000]
        xc = x_t
        if self.noise_concat:
            xc = torch.cat([x_t, z], dim=-1)
        v_pred = self.net(x=xc, t=timesteps, c=z)

        # Compute MSE loss
        loss = F.mse_loss(v_pred, v_target)

        return loss

    @torch.no_grad()
    def forward(self, z, schedule="linear", cfg=1.0, cfg_interval=None):

        b, n, c_z = z.shape

        # Apply NerfEmbedder to condition tokens
        z = self.nerf_embedder(z)

        z = z.reshape(b * n, c_z)
        sample_steps = self.num_sampling_steps

        # get all timesteps ts and intervals Δts
        if schedule == "linear":
            ts = torch.arange(1, sample_steps + 1).flip(0) / sample_steps
            dts = torch.ones_like(ts) * (1.0 / sample_steps)
        elif schedule.startswith("pow"):  # "pow_0.25"
            p = float(schedule.split("_")[1])
            ts = torch.arange(0, sample_steps + 1).flip(0) ** (
                1 / p
            ) / sample_steps ** (1 / p)
            dts = ts[:-1] - ts[1:]
        else:
            raise NotImplementedError
        ts = 1 - ts

        # cfg interval
        if cfg_interval is None:  # cfg_interval = "(.17,1.02)"
            interval = None
        else:
            cfg_lo, cfg_hi = ast.literal_eval(cfg_interval)
            interval = self._edm_to_flow_convention(
                cfg_lo
            ), self._edm_to_flow_convention(cfg_hi)

        # sampling (sample_steps) steps: noise X0 -> clean X1
        trajs = []
        x = torch.randn(b * n, self.in_channels).cuda()  # noise start [b,n,c]
        x = x.to(z.dtype)

        null_z = z.clone() * 0.0 if cfg != 1.0 else None
        for i, (t, dt) in enumerate((zip(ts, dts))):
            timesteps = torch.tensor([t] * (b * n)).to(z.device)

            xc = x
            if self.noise_concat:
                xc = torch.cat([x, z], dim=-1)  # c: 192 + 768 = 960
            vc = self.net(x=xc, t=1000 * timesteps, c=z)  # conditional v

            # classifier free guidance
            if null_z is not None and (
                interval is None
                or ((t.item() >= interval[0]) and (t.item() <= interval[1]))
            ):
                xu = x
                if self.noise_concat:
                    xu = torch.cat([x, null_z], dim=-1)  # c: 192 + 768=960
                vu = self.net(x=xu, t=1000 * timesteps, c=null_z)  # unconditional v
                vc = vu + cfg * (vc - vu)

            # update x
            x = x + dt * vc
            trajs.append(x)

        sampled_token = trajs[-1]
        sampled_image = l2p_transform_tensor(
            sampled_token.reshape(b, n, self.in_channels),
            patch_size=self.patch_size,
        )
        return sampled_image


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            self.mlp[0].weight.dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


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
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
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
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

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

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)


class SpatialGatedFusion(nn.Module):
    """
    修改后的 SimpleMLPAdaLN：
    1. 移除了 TimestepEmbedder 和 t 输入
    2. 专门用于 Token-wise 的空间门控融合
    """

    def __init__(
        self,
        in_channels,  # 语义特征通道数
        model_channels,  # 内部隐藏层维度
        out_channels,  # 输出通道数
        z_channels,  # 细节/条件特征通道数
        num_res_blocks,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        # 【改动1】移除 self.time_embed

        # 映射条件 (细节特征) 到模型维度
        self.cond_embed = nn.Linear(z_channels, model_channels)

        # 映射输入 (语义特征) 到模型维度
        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            # 复用你原有的 ResBlock，无需改动
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)
        # 复用你原有的 FinalLayer，无需改动
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # 【改动2】移除 time_embed 的初始化

        # 保持 Zero-Init 逻辑不变，这对你的“语义优先”策略至关重要
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, c):
        """
        Args:
            x: [B, N, in_channels] - 语义特征 (Main stream)
            c: [B, N, z_channels]  - 细节特征 (Condition stream)
        注意：这里移除了 t 参数
        """
        x = self.input_proj(x)  # [B, N, model_channels]
        c = self.cond_embed(c)  # [B, N, model_channels]

        # 【改动3】不再叠加时间步，直接用 c 作为 condition
        # 因为 c 是 [B, N, C]，传入 ResBlock 后会进行 Token-wise 的自适应调节
        y = c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)


#############################################################
#                 UniFlowVisionModel
#############################################################


class UniFlowVisionModel(PreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = UniFlowVisionConfig

    def __init__(self, config: UniFlowVisionConfig):
        super().__init__(config)
        self.config = config
        vit_hidden_size = config.vit_hidden_size
        llm_hidden_size = config.llm_hidden_size
        self.use_disp_loss = config.use_disp_loss

        # vit encoder
        self.embeddings = UniFlowVisionEmbeddings(config)
        self.encoder = UniFlowVisionEncoder(config)

        self.fusion_layers = [4]
        self.fusion_module = SpatialGatedFusion(
            in_channels=vit_hidden_size,
            model_channels=vit_hidden_size,
            out_channels=vit_hidden_size,
            z_channels=vit_hidden_size,
            num_res_blocks=6,  # 轻量级融合
        )

        # 3. 【新增】Unified Blocks (融合后的深层处理)
        # 你要求的 3 层 Block，用于充分混合语义和细节
        self.unified_depth = 3
        self.sem_blocks = nn.ModuleList(
            [
                Block(
                    dim=vit_hidden_size,
                    num_heads=16,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    init_values=0.0,
                )
                for _ in range(self.unified_depth)
            ]
        )
        self.gen_blocks = nn.ModuleList(
            [
                Block(
                    dim=vit_hidden_size,
                    num_heads=16,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                    init_values=0.0,
                )
                for _ in range(self.unified_depth)
            ]
        )
        # chal.proj, chal.unporj
        self.use_chal_proj = config.use_chal_proj
        self.latent_ch = config.latent_ch
        if self.use_chal_proj:
            # down project to latent_size
            self.chal_proj = nn.Sequential(
                OrderedDict(
                    [
                        ("c_fc", nn.Linear(vit_hidden_size, vit_hidden_size)),
                        ("gelu", nn.GELU()),
                        ("c_proj", nn.Linear(vit_hidden_size, self.latent_ch)),
                    ]
                )
            )
            # up project to hidden_size
            self.chal_unproj = nn.Sequential(
                OrderedDict(
                    [
                        ("c_fc", nn.Linear(self.latent_ch, vit_hidden_size)),
                        ("gelu", nn.GELU()),
                        ("c_proj", nn.Linear(vit_hidden_size, vit_hidden_size)),
                    ]
                )
            )

        # global transformer blocks
        self.global_blocks_depth = config.global_blocks_depth
        self.global_block_pos_embed = nn.Parameter(
            torch.randn(1, self.embeddings.num_patches, vit_hidden_size)
        )
        self.global_blocks = nn.ModuleList(
            [
                FlattenDiTBlock(
                    hidden_size=vit_hidden_size,
                    groups=16,
                    mlp_ratio=4.0,
                )
                for _ in range(self.global_blocks_depth)
            ]
        )
        # token-level flow head
        self.flow_head = FlowDecoder(
            target_channels=3 * config.patch_size * config.patch_size,
            z_channels=config.vit_hidden_size,
            width=config.vit_hidden_size,
            depth=config.num_decoder_layers,
            num_sampling_steps=config.num_sampling_steps,
            grad_checkpointing=False,
            patch_size=config.patch_size,
            img_size=config.image_size,
            use_cfg=config.use_cfg,
            max_freqs=8,
        )

        # init params
        logger.info("Init pos_embed from sincos pos_embed")
        pos_embed_spatial = get_2d_sincos_pos_embed(
            self.global_block_pos_embed.shape[-1],
            int(self.embeddings.num_patches**0.5),  # height or weight
        )
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed_spatial).float())
        self.global_block_pos_embed.data.copy_(
            torch.from_numpy(pos_embed_spatial).float()
        )
        self.apply(self._init_weights)

        # Initialize RoPE position cache for FlattenDiTBlock
        self.precompute_pos = dict()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {}

    @lru_cache
    def fetch_pos(self, height, width, device):
        """Fetch or compute RoPE position embeddings for given spatial dimensions"""
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            # Compute position embeddings based on head_dim
            head_dim = self.config.vit_hidden_size // 16  # num_heads=16
            pos = precompute_freqs_cis_2d(head_dim, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def forward_condition(self, x, return_distill_loss=False):
        """
        Encode images -> Split into Semantic & Gen branches.
        """
        assert x.ndim == 4, f'wrong pixel_values size: {x.shape}'

        # 1. Norm & Embed
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x * 0.5 + 0.5)
        x = self.embeddings(x)

        # 2. Extract Features
        encoder_outputs = self.encoder(
            inputs_embeds=x,
            output_hidden_states=True,
        )
        feat_low = encoder_outputs.hidden_states[self.fusion_layers[0]][:, 1:]
        feat_high = encoder_outputs.last_hidden_state[:, 1:]

        # 3. 门控融合 -> z
        z = self.fusion_module(x=feat_high, c=feat_low)

        # 4. 【分叉】Y-Branch Processing

        # --- 分支 A: 语义对齐 (Semantic Branch) ---
        sem_feat = z
        for block in self.sem_blocks:
            sem_feat = block(sem_feat)  # 迭代更新 sem_feat

        # --- 分支 B: 像素生成 (Generation Branch) ---
        gen_feat = z
        # 【修正】这里必须用 gen_blocks，不要用 sem_blocks
        for block in self.gen_blocks:
            gen_feat = block(gen_feat)  # 迭代更新 gen_feat

        # 5. 后续处理 (仅针对生成分支 gen_feat)
        if self.use_chal_proj:
            gen_feat = self.chal_unproj(self.chal_proj(gen_feat))

        # 6. Global Blocks (RoPE)
        B, N, C = gen_feat.shape
        gen_feat = gen_feat + self.global_block_pos_embed[:, :N]

        grid = int(N**0.5)
        pos = self.fetch_pos(grid, grid, gen_feat.device)

        for block in self.global_blocks:
            gen_feat = block(gen_feat, pos)

        # 7. 返回逻辑
        if return_distill_loss:
            # 计算语义 Loss:
            # 比较 (语义分支的输出 sem_feat) vs (原始高层语义 feat_high)
            distill_loss = F.mse_loss(sem_feat, feat_high.detach())

            # 返回: (用于生成的特征, 蒸馏Loss)
            return gen_feat, distill_loss
        else:
            # 推理时只返回生成特征
            return gen_feat

    def forward_loss(self, target_pixel_values):
        # 1. Get Condition & Distill Loss
        z, distill_loss = self.forward_condition(
            target_pixel_values, return_distill_loss=True
        )

        # 2. Prepare Target
        t = p2l_transform_tensor(target_pixel_values, self.config.patch_size)

        # 3. Compute Flow Loss
        flow_loss = self.flow_head.forward_train(x1=t, z=z)

        # 4. Total Loss
        # 可以添加权重系数，例如: flow_loss + 0.5 * distill_loss
        return {
            'loss': flow_loss + distill_loss,
            'flow_loss': flow_loss,
            'distill_loss': distill_loss,
        }

    def forward(self, pixel_values):
        # Inference: return_distill_loss=False
        z = self.forward_condition(pixel_values, return_distill_loss=False)
        return self.flow_head(z=z)
