import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from src.diffusion.base.training import BaseTrainer


class VAEGANTrainer(BaseTrainer):
    """
    Trainer for VAE with GAN loss.
    Combines reconstruction loss, perceptual loss, and adversarial loss.
    """

    def __init__(self, loss_module: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_module = loss_module

    def _impl_trainstep(
        self, vae_model, vae, solver, x, condition=None, metadata=None
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for VAE model.

        Args:
            vae_model: The integrated VAE model (encoder + decoder)
            vae: Not used (decoder is inside vae_model, kept for compatibility)
            solver: Not used in this trainer (kept for compatibility)
            x: Input images [B, C, H, W] in range [-1, 1]
            condition: Not used (kept for compatibility)
            metadata: Additional metadata

        Returns:
            Dictionary of losses
        """
        # Get global step from metadata if available
        global_step = metadata.get('global_step', 0) if metadata is not None else 0

        # Get student features (from trainable encoder)
        student_features = vae_model.extract_feature(x)

        # Encode image to latent space
        predicted_latents = vae_model.encode_latent(x)

        # Decode latent to reconstruct image
        reconstructed_pixels = vae_model.decode_latent(predicted_latents)

        # Prepare extra result dict with student features
        # Teacher features will be extracted inside loss_module
        extra_result_dict = {
            "student_features": student_features,
        }

        # Compute generator loss (reconstruction + perceptual + GAN)
        total_loss, loss_dict = self.loss_module(
            inputs=x,
            reconstructions=reconstructed_pixels,
            extra_result_dict=extra_result_dict,
            global_step=global_step,
            mode="generator",
        )

        # Convert loss_dict to regular dict with "loss" key for compatibility
        output_dict = {"loss": total_loss}
        output_dict.update(loss_dict)

        return output_dict

    def discriminator_step(
        self, vae_model, vae, x, metadata=None
    ) -> Dict[str, torch.Tensor]:
        """
        Training step for discriminator.

        Args:
            vae_model: The integrated VAE model (encoder + decoder)
            vae: Not used (decoder is inside vae_model, kept for compatibility)
            x: Input images [B, C, H, W] in range [-1, 1]
            metadata: Additional metadata

        Returns:
            Dictionary of discriminator losses
        """
        # Get global step from metadata if available
        global_step = metadata.get('global_step', 0) if metadata is not None else 0

        with torch.no_grad():
            # Encode and decode to get reconstructions
            predicted_latents = vae_model.encode_latent(x)
            reconstructed_pixels = vae_model.decode_latent(predicted_latents)

        # Compute discriminator loss
        discriminator_loss, loss_dict = self.loss_module(
            inputs=x,
            reconstructions=reconstructed_pixels,
            extra_result_dict={},
            global_step=global_step,
            mode="discriminator",
        )

        # Convert to output format
        output_dict = {"discriminator_loss": discriminator_loss}
        output_dict.update(loss_dict)

        return output_dict

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Save state dict"""
        if destination is None:
            destination = {}
        self.loss_module.state_dict(
            destination=destination, prefix=prefix + "loss_module.", keep_vars=keep_vars
        )
        return destination
