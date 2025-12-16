import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import AutoModel, AutoConfig
from diffusers.models import AutoencoderDC
from src.models.transformer.configuration_internvl_chat import InternVLChatConfig
from src.models.transformer.modeling_intern_vit import InternVisionModel
from src.models.transformer.dit_t2i_DeCo import LatentConnectorModule


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
        self.latent_projector = LatentConnectorModule(
            hidden_size=llm_hidden_size, out_channels=self.latent_channel
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
        )
        self.scaling_factor = self.decoder.config.scaling_factor

        # Freeze decoder
        for param in self.decoder.parameters():
            param.requires_grad = False

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
        model = AutoModel.from_pretrained(
            pretrained_model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Extract vision_model and mlp1
        self.vision_model.load_state_dict(model.vision_model.state_dict())
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

    def encode_latent(self, x):
        """
        Encode image to latent representation.
        :param x: input image [B, C, H, W] in range [-1, 1]
        :return: latent [B, latent_channel, H', W']
        """
        # Extract features [B, N, hidden_size]
        features = self.extract_feature(x)

        # Project to latent space [B, N, latent_channel]
        latent = self.latent_projector(features)

        # Layer normalization
        latent = F.layer_norm(latent, normalized_shape=latent.shape[2:], eps=1e-6)

        # Reshape to spatial format [B, latent_channel, H', W']
        grid_size = int(latent.shape[1] ** 0.5)
        latent = latent.transpose(1, 2).reshape(
            latent.shape[0], self.latent_channel, grid_size, grid_size
        )

        return latent

    def decode_latent(self, latent):
        """
        Decode latent to reconstructed image.
        :param latent: latent representation [B, latent_channel, H', W']
        :return: reconstructed image [B, C, H, W] in range [-1, 1]
        """
        # Decode using AutoencoderDC
        reconstructed_pixels = self.decoder.decode(latent).sample

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

    def forward(self, x):
        """
        Full forward pass: encode then decode.
        :param x: input image [B, C, H, W] in range [-1, 1]
        :return: reconstructed image [B, C, H, W] in range [-1, 1]
        """
        latent = self.encode_latent(x)
        reconstructed = self.decode_latent(latent)
        return reconstructed
