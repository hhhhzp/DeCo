from typing import Callable, Iterable, Any, Optional, Union, Sequence
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import AutoModel, AutoConfig
from src.models.autoencoder.base import fp2uint8

# Log to wandb
import wandb
import numpy as np


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.mlp = nn.Sequential(
            nn.LayerNorm(channels, eps=1e-6),
            nn.Linear(channels, channels, bias=True),
            nn.GELU(),
            nn.Linear(channels, channels, bias=True),
        )

    def forward(self, x):
        return x + self.mlp(x)


class DCAE_Decoder(nn.Module):
    """
    DCAE Decoder for pixel reconstruction from vision features.
    """

    def __init__(self, vae, llm_hidden_size):
        super().__init__()
        self.decoder = vae.decoder
        # map clip feature to vae dim
        down_blocks = []
        for i in range(3):
            down_blocks.append(
                ResBlock(
                    llm_hidden_size,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)
        self.down_mlp = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

    def forward(self, vit_embeds):
        for block in self.down_blocks:
            vit_embeds = block(vit_embeds)
        vit_embeds = self.down_mlp(vit_embeds)

        vit_embeds = vit_embeds.permute(0, 2, 1).contiguous()

        b, c, hw = vit_embeds.shape
        z = vit_embeds.view(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))
        h = self.decoder(z)
        return h

    def clip_down(self, vit_embeds):
        for block in self.down_blocks:
            vit_embeds = block(vit_embeds)
        vit_embeds = self.down_mlp(vit_embeds)

        vit_embeds = vit_embeds.permute(0, 2, 1).contiguous()

        b, c, hw = vit_embeds.shape
        z = vit_embeds.view(b, c, int(math.sqrt(hw)), int(math.sqrt(hw)))
        return z

    def vae_decode(self, z):
        h = self.decoder(z)
        return h


class LightningDCAEEvalModel(pl.LightningModule):
    """
    Lightning wrapper for DCAE Decoder evaluation.
    This model loads a pretrained vision encoder and DCAE decoder for image reconstruction.
    """

    def __init__(
        self,
        pretrained_model_path: str = "/apdcephfs/share_300000800/datamultimodal/models/InternVL3-2B",
        vae_weight_path: str = None,
        decoder_checkpoint_path: str = None,
        llm_hidden_size: int = 3200,
    ):
        super().__init__()

        self.pretrained_model_path = pretrained_model_path
        self.vae_weight_path = vae_weight_path
        self.decoder_checkpoint_path = decoder_checkpoint_path
        self.llm_hidden_size = llm_hidden_size

        # Will be initialized in configure_model
        self.vision_model = None
        self.pixel_decoder = None

        self._logged_images_count = 0

    def configure_model(self) -> None:
        """Initialize model weights and load pretrained checkpoints"""
        self.trainer.strategy.barrier()

        # Load vision encoder
        if self.global_rank == 0:
            print(f"Loading vision model from {self.pretrained_model_path}")

        config = AutoConfig.from_pretrained(
            self.pretrained_model_path, trust_remote_code=True
        )
        config.vision_config.drop_path_rate = 0.0
        self.vision_model = AutoModel.from_pretrained(
            self.pretrained_model_path,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Freeze vision model
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # Load VAE for decoder
        if self.global_rank == 0:
            print(f"Loading VAE from {self.vae_weight_path}")

        from diffusers.models import AutoencoderDC

        vae = AutoencoderDC.from_pretrained(
            "/apdcephfs/share_300000800/datamultimodal/models/Sana_1600M_512px_diffusers/vae",
            torch_dtype=torch.bfloat16,
        )

        # Initialize DCAE Decoder
        self.pixel_decoder = DCAE_Decoder(
            vae, self.vision_model.config.llm_config.hidden_size
        )
        state_dict = torch.load(
            os.path.join(
                "/apdcephfs_sh2/share_300000800/user/leike/interns/zhenpeng/project/UniLIP/transferred_weights/UniLIP-3B-Merged",
                "UniLIP-Pixel-Decoder.pth",
            ),
            map_location='cpu',
        )
        msg = self.pixel_decoder.load_state_dict(state_dict)
        print(msg)
        # Set to eval mode
        self.vision_model.eval()
        self.pixel_decoder.eval()

    def resize_down(self, tensor):
        """Resize down to 28/32 ratio"""
        _, _, h, w = tensor.shape

        # Calculate target size (round to nearest integer)
        target_h = round(h * 28 / 32)
        target_w = round(w * 28 / 32)

        return F.interpolate(
            tensor, size=(target_h, target_w), mode='bilinear', align_corners=True
        )

    def _encode_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent representations using vision encoder.

        Args:
            pixel_values: Input images in range [-1, 1]

        Returns:
            latents: Latent representations
        """
        # Ensure input data type matches model
        with torch.no_grad():
            # Convert [-1, 1] range to [0, 1] range, then apply ImageNet normalization
            pixel_values_01 = (pixel_values + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            vit_embeds = self.vision_model.extract_feature(
                Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(pixel_values_01)
            )
            latents = self.pixel_decoder.clip_down(vit_embeds)
        return latents

    def _decode_latents(
        self, latents: torch.Tensor, output_type: str = "tensor"
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Decode latent representations to images using VAE decoder.

        Args:
            latents: Latent representations
            output_type: Output type, either "tensor" or "latent"

        Returns:
            video_frames: Decoded images
        """
        if output_type != "latent":
            latents = latents.to(self.vision_model.dtype)
            # Decode using VAE decoder
            video_frames = self.pixel_decoder.vae_decode(latents)
            # Resize down to 28/32 ratio
            video_frames = self.resize_down(video_frames)
            video_frames = (video_frames * 0.5 + 0.5).clamp(0, 1)
            video = video_frames
        else:
            video = latents

        if output_type == "numpy":
            return video.float().numpy()
        return video

    def on_validation_start(self) -> None:
        """Prepare for validation"""
        self._logged_images_count = 0

    def on_predict_start(self) -> None:
        """Prepare for prediction"""
        self._logged_images_count = 0

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for DCAE Decoder.
        Reconstructs images and logs comparison visualizations.
        """
        img, _, metadata = batch

        with torch.no_grad():
            # Encode to latents
            latents = self._encode_latents(img)

            # Decode to images
            samples = self._decode_latents(latents, output_type="tensor")

            # Convert back to [-1, 1] range for consistency
            samples = samples * 2.0 - 1.0

            # Log first 6 images comparison (original vs reconstructed)
            if self._logged_images_count < 6:
                num_to_log = min(6 - self._logged_images_count, img.shape[0])
                # Convert images from [-1, 1] to [0, 255] uint8
                original_imgs = fp2uint8(img[:num_to_log])
                reconstructed_imgs = fp2uint8(samples[:num_to_log])

                for i in range(num_to_log):
                    # Convert to numpy format [H, W, C]
                    orig_np = original_imgs[i].cpu().permute(1, 2, 0).numpy()
                    recon_np = reconstructed_imgs[i].cpu().permute(1, 2, 0).numpy()

                    # Concatenate horizontally (left: original, right: reconstructed)
                    combined_np = np.concatenate([orig_np, recon_np], axis=1)

                    # Log to wandb with unique key
                    if self.logger is not None:
                        self.logger.experiment.log(
                            {
                                f"reconstruction/sample_{self._logged_images_count + i}": wandb.Image(
                                    combined_np,
                                    caption=f"Sample {self._logged_images_count + i}: Original (Left) | Reconstructed (Right)",
                                ),
                                "global_step": self.global_step,
                            }
                        )

                self._logged_images_count += num_to_log
        return samples

    def validation_step(self, batch, batch_idx):
        """Validation step - same as prediction"""
        return self.predict_step(batch, batch_idx)
