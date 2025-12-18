import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import AutoModel, AutoConfig
from diffusers.models import AutoencoderDC
from src.models.transformer.configuration_internvl_chat import InternVLChatConfig
from src.models.transformer.modeling_intern_vit import InternVisionModel
from src.models.transformer.dit_t2i_DeCo import LatentConnectorModule


import math

import torch
from torch.distributions import Beta


def l2_norm(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return x / torch.clamp(x.norm(dim=-1, keepdim=True), min=eps)


class PowerSphericalDistribution:
    def __init__(self, mu: torch.Tensor, kappa: torch.Tensor, eps: float = 1e-7):
        self.eps = eps
        self.mu = l2_norm(mu, eps)  # [..., m]
        self.kappa = torch.clamp(kappa, min=0.0)

        self.m = self.mu.shape[-1]
        self.d = self.m - 1
        beta_const = 0.5 * self.d
        self.alpha = self.kappa + beta_const  # [...,]
        self.beta = torch.as_tensor(
            beta_const, dtype=self.kappa.dtype, device=self.kappa.device
        ).expand_as(self.kappa)

    def _log_normalizer(self) -> torch.Tensor:
        # log N_X(κ,d) = -[ (α+β)log 2 + β log π + lgamma(α) - lgamma(α+β) ]
        return (
            -(self.alpha + self.beta) * math.log(2.0)
            - self.beta * math.log(math.pi)
            - torch.lgamma(self.alpha)
            + torch.lgamma(self.alpha + self.beta)
        )

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        dot = (self.mu * x).sum(dim=-1).clamp(-1.0, 1.0)
        return self._log_normalizer() + self.kappa * torch.log1p(dot)

    def entropy(self) -> torch.Tensor:
        # H = -[ log N_X + κ ( log 2 + ψ(α) - ψ(α+β) ) ]
        return -(
            self._log_normalizer()
            + self.kappa
            * (
                math.log(2.0)
                + (torch.digamma(self.alpha) - torch.digamma(self.alpha + self.beta))
            )
        )

    def kl_to_uniform(self) -> torch.Tensor:
        # KL(q || U(S^{d})) = -H(q) + log |S^{d}|
        d = torch.as_tensor(self.d, dtype=self.kappa.dtype, device=self.kappa.device)
        log_area = (
            math.log(2.0)
            + 0.5 * (d + 1.0) * math.log(math.pi)
            - torch.lgamma(0.5 * (d + 1.0))
        )
        return -self.entropy() + log_area

    @property
    def mode(self) -> torch.Tensor:
        return self.mu

    def rsample(self):
        Z = Beta(self.alpha, self.beta).rsample()  # [*S, *B]
        t = (2.0 * Z - 1.0).unsqueeze(-1)  # [*S, *B, 1]

        # 2) v ~ U(S^{m-2})
        v = torch.randn(
            *self.mu.shape[:-1],
            self.m - 1,
            device=self.mu.device,
            dtype=self.mu.dtype,
        )  # [*S, *B, m-1]
        v = l2_norm(v, self.eps)

        y = torch.cat(
            [t, torch.sqrt(torch.clamp(1 - t**2, min=0.0)) * v], dim=-1
        )  # [*S, *B, m]

        e1 = torch.zeros_like(self.mu)
        e1[..., 0] = 1.0
        u = l2_norm(e1 - self.mu, self.eps)
        if u.dim() < y.dim():
            u = u.view((1,) * (y.dim() - u.dim()) + u.shape)
        z = y - 2.0 * (y * u).sum(dim=-1, keepdim=True) * u

        parallel = (self.mu - e1).abs().sum(dim=-1, keepdim=True) < 1e-6
        if parallel.any():
            p = parallel
            if p.dim() < y.dim() - 1:
                p = p.view((1,) * (y.dim() - 1 - p.dim()) + p.shape)
            z = torch.where(p, y, z)
        return z


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
        vision_config.num_learnable_tokens = 16
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

        # Latent connector to convert vision features to latent space
        # Output latent_channel + 1 dimensions: latent_channel for mu, 1 for kappa
        self.latent_projector = LatentConnectorModule(
            hidden_size=llm_hidden_size, out_channels=self.latent_channel + 1
        )

        # Load pretrained encoder weights if specified
        if load_pretrained_encoder:
            self.init_vision_model(encoder_config_path)

        # ========== Initialize Decoder ==========
        # Load decoder from pretrained path with subfolder
        self.decoder = AutoencoderDC.from_pretrained(
            decoder_weight_path,
            subfolder=decoder_subfolder,
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
        self.vision_model.load_state_dict(model.vision_model.state_dict(), strict=False)
        self.mlp1.load_state_dict(model.mlp1.state_dict())

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

    def extract_feature(self, pixel_values):
        """
        Extract features from vision model with pixel shuffle downsampling.
        :param pixel_values: input image [B, C, H, W]
        :return: vit_embeds [B, num_patches, hidden_size]
        """
        # Normalize to ImageNet stats
        pixel_values = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(
            pixel_values * 0.5 + 0.5
        )

        vision_model = self.vision_model
        mlp1 = self.mlp1

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
        vit_embeds = mlp1(vit_embeds)
        return vit_embeds

    def normalize(self, x):
        """Normalize latent to unit sphere and scale."""
        x = l2_norm(x)
        x = x * (self.latent_channel**0.5)
        return x

    def encode_latent(self, x, features=None):
        """
        Encode image to Power Spherical distribution.
        :param x: input image [B, C, H, W] in range [-1, 1]
        :param features: pre-extracted features [B, N, hidden_size]
        :return: qz - PowerSphericalDistribution object
        """
        # Extract features [B, N, hidden_size]
        if features is None:
            features = self.extract_feature(x)

        # Project to latent space [B, N, latent_channel + 1]
        latent_params = self.latent_projector(features)

        # Split into mu and kappa
        mu = latent_params[..., :-1]  # [B, N, latent_channel]
        kappa = latent_params[..., -1]  # [B, N]

        # Normalize mu to unit sphere
        mu = l2_norm(mu)
        # Apply softplus to kappa and add 1.0 to ensure kappa > 0
        kappa = F.softplus(kappa) + 1.0

        # Create Power Spherical distribution
        qz = PowerSphericalDistribution(mu, kappa)

        return qz

    def sample_latent(self, qz, use_mode=False):
        """
        Sample latent from Power Spherical distribution and reshape to spatial format.
        :param qz: PowerSphericalDistribution object
        :param use_mode: if True, use mode (mu) instead of sampling; if False, use rsample()
        :return: latent [B, latent_channel, H', W']
        """
        # Sample from distribution (reparameterized) or use mode
        if use_mode:
            latent = qz.mode  # Use mean (deterministic)
        else:
            latent = qz.rsample()  # Random sampling (stochastic)

        # Scale by sqrt(latent_dim)
        latent = latent * (self.latent_channel**0.5)

        # Reshape to spatial format [B, latent_channel, H', W']
        grid_size = int(latent.shape[1] ** 0.5)
        latent = latent.transpose(1, 2).reshape(
            latent.shape[0], self.latent_channel, grid_size, grid_size
        )

        return latent

    def forward_loss(self, qz):
        """
        Compute KL divergence loss for Power Spherical distribution.
        :param qz: PowerSphericalDistribution object
        :return: kl_loss - scalar tensor
        """
        # Compute KL divergence to uniform distribution
        kl_loss = qz.kl_to_uniform()
        return kl_loss.mean()

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

    def forward(self, x, return_features=False, return_kl_loss=False, use_mode=False):
        """
        Full forward pass: encode then decode.
        :param x: input image [B, C, H, W] in range [-1, 1]
        :param return_features: if True, return features in output
        :param return_kl_loss: if True, return KL loss in output
        :param use_mode: if True, use mode (mu) for sampling; if False, use rsample()
        :return: reconstructed image [B, C, H, W] in range [-1, 1]
                 Additional returns based on flags:
                 - if return_kl_loss: (reconstructed, kl_loss)
                 - if return_features: (reconstructed, features)
                 - if both: (reconstructed, features, kl_loss)
        """
        features = self.extract_feature(x)
        qz = self.encode_latent(x, features=features)
        latent = self.sample_latent(qz, use_mode=use_mode)
        reconstructed = self.decode_latent(latent)

        if return_features and return_kl_loss:
            kl_loss = self.forward_loss(qz)
            return reconstructed, features, kl_loss
        elif return_kl_loss:
            kl_loss = self.forward_loss(qz)
            return reconstructed, kl_loss
        elif return_features:
            return reconstructed, features
        return reconstructed
