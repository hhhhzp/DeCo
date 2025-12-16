import torch
from src.models.autoencoder.base import BaseAE


class DCLatentAE(BaseAE):
    """
    DC Autoencoder wrapper for diffusers AutoencoderDC.
    Supports loading from pretrained models with subfolder specification.
    """

    def __init__(
        self, precompute=False, weight_path: str = None, subfolder: str = "vae"
    ):
        super().__init__()
        self.precompute = precompute
        self.model = None
        self.weight_path = weight_path
        self.subfolder = subfolder

        # Import AutoencoderDC from diffusers
        try:
            from diffusers.models import AutoencoderDC
        except ImportError:
            # Fallback to AutoencoderKL if AutoencoderDC is not available
            from diffusers.models import AutoencoderKL as AutoencoderDC

            print("Warning: AutoencoderDC not found, using AutoencoderKL instead")

        # Load model from pretrained path with subfolder
        setattr(
            self,
            "model",
            AutoencoderDC.from_pretrained(
                self.weight_path, subfolder=self.subfolder, torch_dtype=torch.bfloat16
            ),
        )
        self.scaling_factor = self.model.config.scaling_factor

    def _impl_encode(self, x):
        assert self.model is not None
        if self.precompute:
            return x.mul_(self.scaling_factor)
        encodedx = self.model.encode(x).latent_dist.sample().mul_(self.scaling_factor)
        return encodedx

    def _impl_decode(self, x):
        assert self.model is not None
        if self.precompute:
            return x.div_(self.scaling_factor)
        decodedx = self.model.decode(x.div_(self.scaling_factor)).sample
        return decodedx
