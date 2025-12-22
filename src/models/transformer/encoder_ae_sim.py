import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import AutoModel, AutoConfig
from diffusers.models import AutoencoderDC
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from src.models.transformer.configuration_internvl_chat import InternVLChatConfig
from src.models.transformer.modeling_intern_vit import InternVisionModel
from src.models.transformer.dit_t2i_DeCo import LatentConnectorModule
from src.models.layers.rmsnorm import RMSNorm as Norm


class DCDownsampleMLP(nn.Module):
    """
    MLP projection module with shortcut channel grouping (no spatial downsampling).

    Key design:
    - Input: [B, N, in_channels] where N = H*W (spatial tokens)
    - Main path: Channel projection via Linear layer
    - Shortcut path: Channel grouping and averaging (from DCDownBlock2d)
    - MLP with residual connection: x + MLP(x), last layer initialized to 0
    - Output: [B, N, out_channels] - same N as input, ensuring spatial correspondence

    Note: Spatial downsampling is removed as it's already done in VAE model's feature extraction.
          Only channel transformation via shortcut grouping is preserved.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
    ) -> None:
        super().__init__()

        self.shortcut = shortcut
        self.out_channels = out_channels
        self.group_size = 2

        # Channel projection layer (main path)
        self.channel_proj = nn.Linear(in_channels, out_channels)

        # MLP with residual connection
        self.mlp = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.Linear(out_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with channel projection and shortcut grouping.
        :param hidden_states: [B, N, in_channels] sequence format
        :return: [B, N, out_channels] sequence format (N unchanged)
        """
        # Main path: Project channels [B, N, in_channels] -> [B, N, out_channels]
        x = self.channel_proj(hidden_states)

        # Shortcut path: Channel grouping and averaging
        if self.shortcut:
            # [B, N, in_channels] -> [B, N, out_channels, group_size] -> [B, N, out_channels]
            y = hidden_states.unflatten(-1, (self.out_channels, self.group_size))
            y = y.mean(dim=-1)
            x = x + y

        return x + self.mlp(x)


def layer_norm_2d(input: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply layer normalization in 2D spatial format."""
    # input.shape = (bsz, c, h, w)
    _input = input.permute(0, 2, 3, 1)
    _input = F.layer_norm(_input, _input.size()[-1:], None, None, eps)
    _input = _input.permute(0, 3, 1, 2)
    return _input


class VAEModel(nn.Module):
    """
    Integrated VAE model with encoder and decoder.
    Encoder: vision_model + mlp1 + latent_projector
    Decoder: AutoencoderDC from diffusers
    """

    def __init__(
        self,
        encoder_config_path=None,
        decoder_weight_path=None,
        decoder_subfolder="vae",
        select_layer=-1,
        latent_channel=64,
        load_pretrained_encoder=True,
        num_learnable_tokens=0,
    ):
        super().__init__()
        self.select_layer = select_layer
        self.latent_channel = latent_channel
        self.encoder_config_path = encoder_config_path
        self.decoder_weight_path = decoder_weight_path
        self.decoder_subfolder = decoder_subfolder

        # ========== Initialize Encoder ==========
        # Vision encoder
        config = InternVLChatConfig.from_pretrained(encoder_config_path)
        vision_config = config.vision_config
        vision_config.drop_path_rate = 0.0
        vision_config.attention_dropout = 0.0
        vision_config.dropout = 0.0
        vision_config.num_learnable_tokens = num_learnable_tokens
        self.vision_model = InternVisionModel(vision_config)

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        self.downsample_ratio = 0.5

        # MLP to project vision features
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                llm_hidden_size,
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        # self.semantic_projector = nn.Identity()
        # LatentConnectorModule(
        #     hidden_size=llm_hidden_size, out_channels=llm_hidden_size
        # )

        # gen_mlp1: MLP projection with residual connection (no downsampling)
        # Input: vit_hidden_size * 4 channels -> channel projection -> 2*vit_hidden_size (output)
        # Note: Output is 2*vit_hidden_size, NOT llm_hidden_size
        # Spatial dimensions are preserved: [B, N, C] -> [B, N, 2*vit_hidden_size]
        # MLP uses residual connection with last layer initialized to 0
        # Downsampling is already done in extract_vision_features via pixel_shuffle
        self.gen_mlp1 = DCDownsampleMLP(
            in_channels=vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
            out_channels=2 * vit_hidden_size,
        )

        # Latent connector to convert vision features to latent space
        # Output latent_channel dimensions directly (deterministic latent)
        # Input: 2*vit_hidden_size (from gen_mlp1)
        self.latent_projector = LatentConnectorModule(
            hidden_size=2 * vit_hidden_size, out_channels=self.latent_channel
        )

        # Encoder normalization flag (optional)
        self.encoder_norm = False

        # Load pretrained encoder weights if specified
        if load_pretrained_encoder:
            self.init_vision_model(encoder_config_path)

        # ========== Initialize Decoder ==========
        # Load decoder from pretrained path with subfolder
        self.decoder = AutoencoderDC.from_pretrained(
            decoder_weight_path,
            # subfolder=decoder_subfolder,
            torch_dtype=torch.bfloat16,
        ).decoder
        # self.scaling_factor = self.decoder.config.scaling_factor

    def init_vision_model(self, pretrained_model_path: str):
        """
        Load vision_model and mlp1 from pretrained InternVLChatModel.

        Args:
            pretrained_model_path: Path to pretrained model
        """
        print(f"Loading pretrained encoder from {pretrained_model_path}...")

        # Load pretrained InternVLChatModel
        config = AutoConfig.from_pretrained(
            pretrained_model_path, trust_remote_code=True
        )
        config.vision_config.drop_path_rate = 0.0
        config.vision_config.attention_dropout = 0.0
        config.vision_config.dropout = 0.0
        model = AutoModel.from_pretrained(
            pretrained_model_path,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Extract vision_model and mlp1
        self.vision_model = model.vision_model  # .state_dict(), strict=False)
        self.mlp1 = model.mlp1  # .state_dict())

        print("Pretrained encoder weights loaded successfully!")

    def pixel_shuffle(self, x, scale_factor=0.5):
        """Pixel shuffle downsampling"""
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _process_vit_features(self, vit_embeds):
        """
        Process raw ViT features: remove CLS token, reshape, pixel shuffle, and flatten.
        :param vit_embeds: [B, num_patches+1, hidden_size] raw ViT output
        :return: [B, N, hidden_size*4] processed features
        """
        vit_embeds = vit_embeds[:, 1:, :]  # Remove CLS token

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        return vit_embeds

    def extract_vision_features(self, pixel_values, select_layer=None):
        """
        Extract raw features from vision model (before MLP projection).
        :param pixel_values: input image [B, C, H, W]
        :param select_layer: which layer to extract features from (if None, use self.select_layer)
        :return: vit_embeds [B, num_patches, vit_hidden_size * 4] (after pixel shuffle, before MLP)
        """
        # Normalize to ImageNet stats
        pixel_values = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(
            pixel_values * 0.5 + 0.5
        )

        vision_model = self.vision_model
        layer_idx = select_layer if select_layer is not None else self.select_layer

        if layer_idx == -1:
            vit_embeds = vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[layer_idx]

        return self._process_vit_features(vit_embeds)

    def extract_feature(self, pixel_values, use_gen_mlp=False, vision_features=None):
        """
        Extract features from vision model with pixel shuffle downsampling.
        :param pixel_values: input image [B, C, H, W]
        :param use_gen_mlp: if True, use gen_mlp1 (output: 2*vit_hidden_size) with layer 18;
                           if False, use mlp1 (output: llm_hidden_size) with last layer
        :param vision_features: pre-computed vision features (before MLP), if provided, skip vision model computation
        :return: vit_embeds [B, N, hidden_size]
                 - if use_gen_mlp=False: [B, N, llm_hidden_size]
                 - if use_gen_mlp=True: [B, N, 2*vit_hidden_size]
                 Both have the same N (spatial correspondence guaranteed)
        """
        # Extract vision features if not provided
        if vision_features is None:
            # Select different layers based on use_gen_mlp
            select_layer = 18 if use_gen_mlp else -1
            vision_features = self.extract_vision_features(
                pixel_values, select_layer=select_layer
            )

        # Apply MLP projection
        # Both mlp1 and gen_mlp1 preserve spatial dimensions (N unchanged)
        mlp = self.gen_mlp1 if use_gen_mlp else self.mlp1
        vit_embeds = mlp(vision_features)
        return vit_embeds

    def encode_latent(self, x, features=None, return_dict=True):
        """
        Encode image to deterministic latent representation.
        Uses gen_mlp1 for feature extraction (latent mode).
        :param x: input image [B, C, H, W] in range [-1, 1]
        :param features: pre-extracted features [B, N, hidden_size]
        :param return_dict: if True, return AutoencoderKLOutput; else return tuple (for compatibility)
        :return: latent [B, latent_channel, H', W'] or AutoencoderKLOutput with latent_dist=latent
        """
        # Extract features [B, N, hidden_size] using gen_mlp1
        if features is None:
            features = self.extract_feature(x, use_gen_mlp=True)

        # 2. Spatial normalization on encoder features [B, T, D]
        # gamma = 0.6  # normalization strength
        # features = features - gamma * features.mean(dim=1, keepdim=True)
        # features = features / (features.std(dim=1, keepdim=True) + 1e-6)

        # Project to latent space [B, N, latent_channel]
        latent = self.latent_projector(features)

        # Reshape to spatial format [B, latent_channel, H', W']
        grid_size = int(latent.shape[1] ** 0.5)
        latent = latent.transpose(1, 2).reshape(
            latent.shape[0], self.latent_channel, grid_size, grid_size
        )

        # Optional: apply layer normalization
        # if self.encoder_norm:
        #     latent = layer_norm_2d(latent)

        if not return_dict:
            return (latent,)

        # Return in AutoencoderKLOutput format for compatibility
        # Store latent directly in latent_dist field
        return latent

    def sample_latent(self, latent_or_output, use_mode=False):
        """
        Extract latent from output (for backward compatibility).
        :param latent_or_output: latent tensor or AutoencoderKLOutput
        :param use_mode: deprecated parameter (kept for interface compatibility)
        :return: latent [B, latent_channel, H', W']
        """
        # Extract latent if wrapped in AutoencoderKLOutput
        if isinstance(latent_or_output, AutoencoderKLOutput):
            return latent_or_output.latent_dist

        # Otherwise return as-is
        return latent_or_output

    def decode_latent(self, latent):
        """
        Decode latent to reconstructed image.
        :param latent: latent representation [B, latent_channel, H', W']
        :return: reconstructed image [B, C, H, W] in range [-1, 1]
        """
        # Decode using AutoencoderDC
        reconstructed_pixels = self.decoder(latent)

        # Scale spatial dimensions to 14/16 of original size
        scale_factor = 14.0 / 16.0
        B, C, H, W = reconstructed_pixels.shape
        target_H = int(H * scale_factor)
        target_W = int(W * scale_factor)

        reconstructed_pixels = F.interpolate(
            reconstructed_pixels,
            size=(target_H, target_W),
            mode='bilinear',
            align_corners=False,
        )

        return reconstructed_pixels

    def forward(
        self,
        x,
        return_features=False,
        use_mode=False,
    ):
        """
        Full forward pass: encode then decode.
        Supports two output modes:
        1. Feature mode: vision_model + mlp1 -> features (last layer)
        2. Latent mode: vision_model + gen_mlp1 + latent_projector -> latent -> reconstructed (layer 18)

        :param x: input image [B, C, H, W] in range [-1, 1]
        :param return_features: if True, return features in output
        :param use_mode: deprecated parameter (kept for interface compatibility)
        :return: reconstructed image [B, C, H, W] in range [-1, 1]
                 If return_features=True: (reconstructed, features)
        """
        # Normalize to ImageNet stats (shared preprocessing)
        pixel_values = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(
            x * 0.5 + 0.5
        )

        # Forward vision model once with output_hidden_states=True to get all layers
        vision_outputs = self.vision_model(
            pixel_values=pixel_values, output_hidden_states=True, return_dict=True
        )

        # Extract layer 18 features for gen_mlp1 (latent encoding)
        gen_features = self._process_vit_features(vision_outputs.hidden_states[18])
        gen_features = self.gen_mlp1(gen_features)
        latent = self.encode_latent(x, features=gen_features)
        reconstructed = self.decode_latent(latent)

        # If features are requested, extract using mlp1 (feature mode) from last layer
        if return_features:
            features = self._process_vit_features(vision_outputs.last_hidden_state)
            features = self.mlp1(features)
            return reconstructed, features

        return reconstructed
