"""
Custom DDP Strategy for Multi-Model Training (e.g., GAN with Generator and Discriminator)

This strategy wraps multiple sub-models (generator and discriminator) separately in DDP,
avoiding the issue where toggle_optimizer causes gradient tracking problems.

Reference: pytorch_lightning.strategies.ddp.DDPStrategy
"""

from contextlib import nullcontext
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_info
import lightning.pytorch as pl


class MultiModelDDPStrategy(DDPStrategy):
    """
    Custom DDP Strategy that wraps generator and discriminator separately.

    This solves the issue where toggle_optimizer in manual optimization mode
    causes DDP to lose track of which parameters should receive gradients.

    Instead of wrapping the entire LightningModule, we wrap only the sub-models
    that need to be trained (vae_model and discriminator).
    """

    def configure_ddp(self) -> None:
        """
        Override configure_ddp to wrap sub-models separately instead of the entire model.

        Ref pytorch_lightning.strategies.ddp.DDPStrategy:
            def configure_ddp(self) -> None:
                self.model = self._setup_model(self.model)
                self._register_ddp_hooks()
        """
        rank_zero_info(
            f"{self.__class__.__name__}: configuring DistributedDataParallel for sub-models"
        )

        # Get device_ids for DDP
        device_ids = self.determine_ddp_device_ids()
        rank_zero_info(f"Setting up DDP with device ids: {device_ids}")

        # Use CUDA stream context if using GPU
        # https://pytorch.org/docs/stable/notes/cuda.html#id5
        ctx = (
            torch.cuda.stream(torch.cuda.Stream())
            if device_ids is not None
            else nullcontext()
        )

        with ctx:
            # Check if this is a VAE training module with separate generator/discriminator
            if hasattr(self.model, 'vae_model') and hasattr(self.model, 'loss_module'):
                # Wrap VAE model (generator/encoder) in DDP
                if not isinstance(self.model.vae_model, DDP):
                    rank_zero_info("[MultiModelDDPStrategy] Wrapping vae_model in DDP")
                    self.model.vae_model = DDP(
                        self.model.vae_model,
                        device_ids=device_ids,
                        find_unused_parameters=False,  # VAE encoder should use all parameters
                        **self._ddp_kwargs,
                    )

                # Wrap loss_module (contains discriminator and perceptual_loss) in DDP
                if not isinstance(self.model.loss_module, DDP):
                    rank_zero_info(
                        "[MultiModelDDPStrategy] Wrapping loss_module in DDP"
                    )
                    self.model.loss_module = DDP(
                        self.model.loss_module,
                        device_ids=device_ids,
                        find_unused_parameters=False,  # All components should be used
                        **self._ddp_kwargs,
                    )
            else:
                # Fallback: wrap the entire model if it doesn't match expected structure
                rank_zero_info(
                    "[MultiModelDDPStrategy] Model structure not recognized, wrapping entire model"
                )
                self.model = self._setup_model(self.model)

        # Register DDP hooks
        self._register_ddp_hooks()
