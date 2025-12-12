import lightning.pytorch as pl
from lightning.pytorch import Callback

import torch
import numpy as np
from typing import Any, Dict
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_info


class ComputeMetricsHook(Callback):
    """
    Callback for computing reconstruction quality metrics (PSNR, SSIM).
    Collects original and reconstructed images during validation/prediction,
    then computes and logs metrics at epoch end.
    """

    def __init__(self, metric_batch_size: int = 32):
        """
        Args:
            metric_batch_size: Batch size for processing metrics to avoid OOM
        """
        self.metric_batch_size = metric_batch_size
        self.psnr_values = []
        self.ssim_values = []

    def metrics_start(self):
        """Initialize metric collection"""
        self.psnr_values = []
        self.ssim_values = []
        rank_zero_info("Starting metric computation...")

    def collect_batch(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        reconstructed: STEP_OUTPUT,
        batch: Any,
    ) -> None:
        """
        Compute metrics for each batch independently on each GPU.

        Args:
            trainer: PyTorch Lightning trainer
            pl_module: Lightning module
            reconstructed: Reconstructed images (output from model), float [-1, 1]
            batch: Original batch containing input images, float [-1, 1]
        """
        from skimage.metrics import peak_signal_noise_ratio as psnr_loss
        from skimage.metrics import structural_similarity as ssim_loss

        # Extract original images from batch (range: [-1, 1])
        original_img, _, metadata = batch

        # Convert both images from [-1, 1] to [0, 1]
        reconstructed = (reconstructed + 1.0) / 2.0
        reconstructed = torch.clamp(reconstructed, 0.0, 1.0)

        original_img = (original_img + 1.0) / 2.0
        original_img = torch.clamp(original_img, 0.0, 1.0)

        # Convert to numpy for metric computation
        original_np = original_img.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()

        batch_size = original_np.shape[0]

        # Compute metrics for each image in the batch
        for i in range(batch_size):
            # Convert to [H, W, C] format
            orig = np.transpose(original_np[i], (1, 2, 0))
            recon = np.transpose(reconstructed_np[i], (1, 2, 0))

            # Compute PSNR (data_range=1.0 because images are in [0,1] range)
            psnr_val = psnr_loss(orig, recon, data_range=1.0)
            self.psnr_values.append(psnr_val)

            # Compute SSIM
            ssim_val = ssim_loss(orig, recon, data_range=1.0, channel_axis=-1)
            self.ssim_values.append(ssim_val)

    def metrics_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """
        Aggregate and log metrics at epoch end across all GPUs.
        """
        if len(self.psnr_values) == 0:
            rank_zero_info("No metrics collected")
            return

        # Convert to tensors for distributed synchronization
        psnr_tensor = torch.tensor(self.psnr_values, device=pl_module.device)
        ssim_tensor = torch.tensor(self.ssim_values, device=pl_module.device)

        # Gather all values from all GPUs
        all_psnr = pl_module.all_gather(psnr_tensor).flatten()
        all_ssim = pl_module.all_gather(ssim_tensor).flatten()

        # Only log on rank 0
        if trainer.is_global_zero:
            # Compute statistics
            psnr_mean = all_psnr.mean().item()
            psnr_std = all_psnr.std().item()
            ssim_mean = all_ssim.mean().item()
            ssim_std = all_ssim.std().item()

            rank_zero_info(f"Computed metrics for {len(all_psnr)} images")
            rank_zero_info(f"PSNR: {psnr_mean:.4f} ± {psnr_std:.4f}")
            rank_zero_info(f"SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}")

            # Log to tensorboard/logger
            pl_module.log(
                "metrics/psnr",
                psnr_mean,
                on_epoch=True,
                sync_dist=False,
                rank_zero_only=True,
            )
            pl_module.log(
                "metrics/ssim",
                ssim_mean,
                on_epoch=True,
                sync_dist=False,
                rank_zero_only=True,
            )

        # Clear collected metrics
        self.psnr_values = []
        self.ssim_values = []

    # ========== Validation hooks ==========
    def on_validation_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.metrics_start()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.collect_batch(trainer, pl_module, outputs, batch)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.metrics_end(trainer, pl_module)

    # ========== Prediction hooks ==========
    def on_predict_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.metrics_start()

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.collect_batch(trainer, pl_module, outputs, batch)

    def on_predict_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.metrics_end(trainer, pl_module)

    def state_dict(self) -> Dict[str, Any]:
        return dict()
