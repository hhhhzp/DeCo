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

        return self.mlp(x)


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
        self.semantic_projector = nn.Identity()
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
        # Output 2*latent_channel dimensions: latent_channel for mean, latent_channel for logvar
        # Input: 2*vit_hidden_size (from gen_mlp1)
        self.latent_projector = LatentConnectorModule(
            hidden_size=2 * vit_hidden_size, out_channels=2 * self.latent_channel
        )

        # Encoder normalization flag (optional)
        self.encoder_norm = True
        self.deterministic = False

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

    def copy_mlp1_to_gen_mlp1(self):
        """
        Copy weights from mlp1 to gen_mlp1.
        This initializes gen_mlp1 with the same weights as mlp1.
        """
        self.gen_mlp1.load_state_dict(self.mlp1.state_dict())
        print("Copied mlp1 weights to gen_mlp1")

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

    def extract_vision_features(self, pixel_values):
        """
        Extract raw features from vision model (before MLP projection).
        :param pixel_values: input image [B, C, H, W]
        :return: vit_embeds [B, num_patches, vit_hidden_size * 4] (after pixel shuffle, before MLP)
        """
        # Normalize to ImageNet stats
        pixel_values = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(
            pixel_values * 0.5 + 0.5
        )

        vision_model = self.vision_model

        if self.select_layer == -1:
            vit_embeds = vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]

        vit_embeds = vit_embeds[:, 1:, :]  # Remove CLS token

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        return vit_embeds

    def extract_feature(self, pixel_values, use_gen_mlp=False, vision_features=None):
        """
        Extract features from vision model with pixel shuffle downsampling.
        :param pixel_values: input image [B, C, H, W]
        :param use_gen_mlp: if True, use gen_mlp1 (output: 2*vit_hidden_size); if False, use mlp1 (output: llm_hidden_size)
        :param vision_features: pre-computed vision features (before MLP), if provided, skip vision model computation
        :return: vit_embeds [B, N, hidden_size]
                 - if use_gen_mlp=False: [B, N, llm_hidden_size]
                 - if use_gen_mlp=True: [B, N, 2*vit_hidden_size]
                 Both have the same N (spatial correspondence guaranteed)
        """
        # Extract vision features if not provided
        if vision_features is None:
            vision_features = self.extract_vision_features(pixel_values)

        # Apply MLP projection
        # Both mlp1 and gen_mlp1 preserve spatial dimensions (N unchanged)
        mlp = self.gen_mlp1 if use_gen_mlp else self.mlp1
        vit_embeds = mlp(vision_features)
        if not use_gen_mlp and hasattr(self, "semantic_projector"):
            vit_embeds = self.semantic_projector(vit_embeds)
        return vit_embeds

    def encode_latent(self, x, features=None, return_dict=True):
        """
        Encode image to Diagonal Gaussian distribution with dynamic spatial alignment.
        Ensures latent size matches Decoder's 32x downsampling requirement.
        """
        # 1. 动态计算目标 Latent 尺寸
        # Decoder stride = 32. 输入 448 -> 14, 512 -> 16
        B, C, H, W = x.shape
        target_H = H // 32
        target_W = W // 32

        # 2. 提取特征 [B, N, C]
        # 使用 gen_mlp1 提取特征
        if features is None:
            features = self.extract_feature(x, use_gen_mlp=True)

        # 3. 空间重组与插值对齐 (关键修正步骤)
        # 获取当前特征的空间维度
        current_N = features.shape[1]
        current_size = int(current_N**0.5)  # 假设正方形特征，例如 16

        # [B, N, C] -> [B, C, H_curr, W_curr]
        features_spatial = features.transpose(1, 2).reshape(
            B, -1, current_size, current_size
        )

        # 如果当前尺寸与目标 Latent 尺寸不一致，进行双线性插值
        # 例如：从 16x16 (Encoder输出) 插值到 14x14 (Decoder需求)
        if (current_size, current_size) != (target_H, target_W):
            features_spatial = F.interpolate(
                features_spatial,
                size=(target_H, target_W),
                mode='bilinear',
                align_corners=False,
            )

        # 插值后展平回序列格式: [B, C, H_target, W_target] -> [B, N_target, C]
        features = features_spatial.flatten(2).transpose(1, 2)

        # 4. 投影到 Latent 分布参数空间
        # moments shape: [B, N_target, 2*latent_channel]
        moments = self.latent_projector(features)

        # 5. 重塑为 VAE 标准空间格式 [B, 2*latent_channel, H_target, W_target]
        moments = moments.transpose(1, 2).reshape(
            moments.shape[0], 2 * self.latent_channel, target_H, target_W
        )

        # 6. 拆分 Mean 和 Logvar
        mean, logvar = torch.chunk(moments, 2, dim=1)

        # 7. 可选归一化 (注意：这可能会限制分布范围，调试时需留意)
        if self.encoder_norm:
            mean = layer_norm_2d(mean)

        # 8. 重新拼接并构建分布
        moments = torch.cat([mean, logvar], dim=1).contiguous()

        posterior = DiagonalGaussianDistribution(
            moments, deterministic=self.deterministic
        )

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def sample_latent(self, posterior, use_mode=False):
        """
        Sample latent from Diagonal Gaussian distribution.
        :param posterior: DiagonalGaussianDistribution object or AutoencoderKLOutput
        :param use_mode: if True, use mode() instead of sampling; if False, use sample()
        :return: latent [B, latent_channel, H', W']
        """
        # Extract distribution if wrapped in AutoencoderKLOutput
        if isinstance(posterior, AutoencoderKLOutput):
            posterior = posterior.latent_dist

        # Sample from distribution or use mode
        if use_mode:
            latent = posterior.mode()  # Use mean (deterministic)
        else:
            latent = posterior.sample()  # Random sampling (stochastic)

        return latent

    def forward_loss(self, posterior):
        """
        Compute KL divergence loss for Diagonal Gaussian distribution.
        :param posterior: DiagonalGaussianDistribution object or AutoencoderKLOutput
        :return: kl_loss - scalar tensor
        """
        # Extract distribution if wrapped in AutoencoderKLOutput
        if isinstance(posterior, AutoencoderKLOutput):
            posterior = posterior.latent_dist

        # Compute KL divergence to standard normal distribution
        kl_loss = posterior.kl()
        return kl_loss.mean()

    def decode_latent(self, latent):
        """
        Decode latent to reconstructed image.
        """
        # Decode using AutoencoderDC
        # latent size 已经对齐了 (e.g. 14x14)，输出直接是 448x448
        reconstructed_pixels = self.decoder(latent)

        # 原始代码中的 F.interpolate 已被移除
        # 此时 reconstructed_pixels 应该已经是目标分辨率

        return reconstructed_pixels

    def forward(
        self,
        x,
        return_features=False,
        return_kl_loss=False,
        use_mode=False,
    ):
        """
        Full forward pass: encode then decode.
        Supports two output modes:
        1. Feature mode: vision_model + mlp1 -> features
        2. Latent mode: vision_model + gen_mlp1 + latent_projector -> latent -> reconstructed

        :param x: input image [B, C, H, W] in range [-1, 1]
        :param return_features: if True, return features in output
        :param return_kl_loss: if True, return KL loss in output
        :param use_mode: if True, use mode() for sampling; if False, use sample()
        :param sample_posterior: if True, sample from posterior; if False, use mode (same as use_mode)
        :return: reconstructed image [B, C, H, W] in range [-1, 1]
                 Additional returns based on flags:
                 - if return_kl_loss: (reconstructed, kl_loss)
                 - if return_features: (reconstructed, features)
                 - if both: (reconstructed, features, kl_loss)
        """
        # Extract vision features once (before MLP) to avoid redundant computation
        vision_features = self.extract_vision_features(x)

        # Extract features using gen_mlp1 for latent encoding
        gen_features = self.extract_feature(
            x, use_gen_mlp=True, vision_features=vision_features
        )
        posterior = self.encode_latent(x, features=gen_features)

        # Determine sampling mode
        latent = self.sample_latent(posterior, use_mode=use_mode)
        reconstructed = self.decode_latent(latent)

        # If features are requested, extract using mlp1 (feature mode)
        # Reuse the same vision_features to avoid redundant vision model computation
        if return_features:
            features = self.extract_feature(
                x, use_gen_mlp=False, vision_features=vision_features
            )

        if return_features and return_kl_loss:
            kl_loss = self.forward_loss(posterior)
            return reconstructed, features, kl_loss
        elif return_kl_loss:
            kl_loss = self.forward_loss(posterior)
            return reconstructed, kl_loss
        elif return_features:
            return reconstructed, features
        return reconstructed
