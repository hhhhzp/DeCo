from typing import Callable, Iterable, Any, Optional, Union, Sequence, Mapping, Dict
import os.path
import copy
import torch
import torch.nn as nn
import torchvision
import transformers
import lightning.pytorch as pl
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from lightning.pytorch.callbacks import Callback
from transformers import AutoModel, AutoConfig, get_constant_schedule_with_warmup

from src.models.autoencoder.base import BaseAE, fp2uint8
from src.models.transformer.encoder_ae import VAEModel
from src.utils.model_loader import ModelLoader
from src.callbacks.simple_ema import SimpleEMA
from src.diffusion.vae_gan.training_ae import VAEGANTrainer
from src.utils.no_grad import no_grad, filter_nograd_tensors
from src.utils.copy import copy_params

EMACallable = Callable[[nn.Module, nn.Module], SimpleEMA]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]


class LightningModelVAE(pl.LightningModule):
    """
    Lightning module for training VAE encoder with GAN loss.
    Uses integrated VAEModel with encoder and decoder.
    """

    def __init__(
        self,
        vae_model: VAEModel,
        loss_module: nn.Module,
        ema_tracker: SimpleEMA = None,
        optimizer: OptimizerCallable = None,
        lr_scheduler: LRSchedulerCallable = None,
        eval_original_model: bool = False,
        pretrain_model_path: str = None,
        discriminator_optimizer: OptimizerCallable = None,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.ema_vae_model = copy.deepcopy(self.vae_model)

        # Create vae_trainer with loss_module
        self.vae_trainer = VAEGANTrainer(loss_module=loss_module, null_condition_p=0)

        self.ema_tracker = ema_tracker
        self.optimizer = optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.lr_scheduler = lr_scheduler

        self.eval_original_model = eval_original_model
        self.pretrain_model_path = pretrain_model_path

        self._strict_loading = False
        self._logged_images_count = 0

        # Automatic optimization disabled for manual discriminator training
        self.automatic_optimization = False

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()

        # Load pretrained weights if specified
        if self.pretrain_model_path is not None:
            checkpoint = torch.load(self.pretrain_model_path, map_location='cpu')
            msg = self.load_state_dict(checkpoint['state_dict'], strict=False)
            if self.global_rank == 0:
                print(f"Loaded pretrained model from {self.pretrain_model_path}: {msg}")

        # Copy parameters to EMA model
        copy_params(src_model=self.vae_model, dst_model=self.ema_vae_model)

        # Disable grad for decoder (already frozen in VAEModel.__init__)
        # and EMA model
        no_grad(self.vae_model.decoder)
        no_grad(self.ema_vae_model)

        # Compile models for efficiency
        self.vae_model = torch.compile(self.vae_model)
        self.ema_vae_model = torch.compile(self.ema_vae_model)

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.ema_tracker]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Optimizer for encoder (generator)
        params_encoder = filter_nograd_tensors(self.vae_model.parameters())
        optimizer_encoder = self.optimizer({"params": params_encoder})
        lr_scheduler_encoder = get_constant_schedule_with_warmup(
            optimizer_encoder, num_warmup_steps=1000
        )

        # Optimizer for discriminator
        discriminator = self.vae_trainer.loss_module.discriminator
        params_discriminator = filter_nograd_tensors(discriminator.parameters())
        optimizer_discriminator = self.discriminator_optimizer(
            [{"params": params_discriminator}]
        )
        lr_scheduler_discriminator = get_constant_schedule_with_warmup(
            optimizer_discriminator, num_warmup_steps=1000
        )

        return [
            {
                "optimizer": optimizer_encoder,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_encoder,
                    "interval": "step",
                    "frequency": 1,
                },
            },
            {
                "optimizer": optimizer_discriminator,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_discriminator,
                    "interval": "step",
                    "frequency": 1,
                },
            },
        ]

    def on_validation_start(self) -> None:
        self.ema_vae_model.to(torch.float32)
        self._logged_images_count = 0

    def on_predict_start(self) -> None:
        self.ema_vae_model.to(torch.float32)
        self._logged_images_count = 0

    def on_train_start(self) -> None:
        self.ema_vae_model.to(torch.float32)
        self.ema_tracker.setup_models(net=self.vae_model, ema_net=self.ema_vae_model)

    def training_step(self, batch, batch_idx):
        img, _, metadata = batch

        # Get optimizers
        opt_encoder, opt_discriminator = self.optimizers()

        # Add global step to metadata
        metadata = metadata or {}
        metadata['global_step'] = self.global_step

        # ========== Train Generator (Encoder) ==========
        opt_encoder.zero_grad()

        # Forward pass through VAE model
        loss_dict = self.vae_trainer(
            self.vae_model,
            None,  # decoder is inside vae_model
            None,  # solver not used
            img,
            condition=None,
            uncondition=None,
            metadata=metadata,
        )

        # Backward and optimize
        self.manual_backward(loss_dict["loss"])
        # Manual gradient clipping for encoder
        torch.nn.utils.clip_grad_norm_(self.vae_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.vae_trainer.parameters(), max_norm=1.0)
        opt_encoder.step()

        # Update learning rate
        sch_encoder = self.lr_schedulers()[0]
        sch_encoder.step()

        # ========== Train Discriminator ==========
        # Only train discriminator after warmup steps
        if self.vae_trainer.loss_module.should_discriminator_be_trained(
            self.global_step
        ):
            opt_discriminator.zero_grad()

            # Discriminator forward pass
            disc_loss_dict = self.vae_trainer.discriminator_step(
                self.vae_model,
                None,  # decoder is inside vae_model
                img,
                metadata=metadata,
            )

            # Backward and optimize
            self.manual_backward(disc_loss_dict["discriminator_loss"])
            # Manual gradient clipping for discriminator
            torch.nn.utils.clip_grad_norm_(
                self.vae_trainer.loss_module.discriminator.parameters(), max_norm=1.0
            )
            opt_discriminator.step()

            # Update learning rate
            sch_discriminator = self.lr_schedulers()[1]
            sch_discriminator.step()

            # Merge discriminator losses into main loss dict
            loss_dict.update(disc_loss_dict)

        # Log learning rates
        loss_dict["lr_encoder"] = opt_encoder.param_groups[0]['lr']
        loss_dict["lr_discriminator"] = opt_discriminator.param_groups[0]['lr']

        # Log all losses
        self.log_dict(loss_dict, prog_bar=True, on_step=True, sync_dist=False)

        return loss_dict["loss"]

    def predict_step(self, batch, batch_idx):
        img, _, metadata = batch

        # Select model (original or EMA)
        model = self.vae_model if self.eval_original_model else self.ema_vae_model

        with torch.no_grad():
            # Encode to latent and decode to reconstruct
            samples = model(img)

            # Log first 6 images comparison
            if self._logged_images_count < 6:
                num_to_log = min(6 - self._logged_images_count, img.shape[0])

                # Convert images from [-1, 1] to [0, 255] uint8
                original_imgs = fp2uint8(img[:num_to_log])
                reconstructed_imgs = fp2uint8(samples[:num_to_log])

                # Log each comparison image with wandb
                import wandb
                import numpy as np

                for i in range(num_to_log):
                    # Convert to numpy format [H, W, C]
                    orig_np = original_imgs[i].cpu().permute(1, 2, 0).numpy()
                    recon_np = reconstructed_imgs[i].cpu().permute(1, 2, 0).numpy()

                    # Concatenate horizontally (left: original, right: reconstructed)
                    combined_np = np.concatenate([orig_np, recon_np], axis=1)

                    # Log to wandb with unique key
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
        samples = self.predict_step(batch, batch_idx)
        return samples

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.vae_model.state_dict(
            destination=destination, prefix=prefix + "vae_model.", keep_vars=keep_vars
        )
        self.ema_vae_model.state_dict(
            destination=destination,
            prefix=prefix + "ema_vae_model.",
            keep_vars=keep_vars,
        )
        self.vae_trainer.state_dict(
            destination=destination,
            prefix=prefix + "vae_trainer.",
            keep_vars=keep_vars,
        )
        return destination
