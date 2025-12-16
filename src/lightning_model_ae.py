from typing import Callable, Iterable, Any, Optional, Union, Sequence, Mapping, Dict
import os.path
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
from src.utils.no_grad import no_grad, filter_nograd_tensors

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
        optimizer: OptimizerCallable = None,
        lr_scheduler: LRSchedulerCallable = None,
        eval_original_model: bool = False,
        pretrain_model_path: str = None,
        discriminator_optimizer: OptimizerCallable = None,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.loss_module = loss_module

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

        # Print trainable parameters
        if self.global_rank == 0:
            print("\n" + "=" * 80)
            print("TRAINABLE PARAMETERS SUMMARY")
            print("=" * 80)

            # Collect all trainable parameters
            trainable_params = set()
            total_trainable = 0

            # Check VAE model parameters
            print("\n[VAE Model - Encoder]")
            for name, param in self.vae_model.named_parameters():
                if param.requires_grad:
                    param_id = id(param)
                    if param_id not in trainable_params:
                        trainable_params.add(param_id)
                        total_trainable += param.numel()
                        print(f"  ✓ {name}: {param.shape} ({param.numel():,} params)")

            # Check discriminator parameters
            print("\n[Discriminator]")
            for (
                name,
                param,
            ) in self.loss_module.discriminator.named_parameters():
                if param.requires_grad:
                    param_id = id(param)
                    if param_id not in trainable_params:
                        trainable_params.add(param_id)
                        total_trainable += param.numel()
                        print(f"  ✓ {name}: {param.shape} ({param.numel():,} params)")

            # Check other loss module parameters
            print("\n[Loss Module - Other Trainable Components]")
            has_other_trainable = False
            for name, param in self.loss_module.named_parameters():
                if param.requires_grad and 'discriminator' not in name:
                    param_id = id(param)
                    if param_id not in trainable_params:
                        trainable_params.add(param_id)
                        total_trainable += param.numel()
                        print(f"  ✓ {name}: {param.shape} ({param.numel():,} params)")
                        has_other_trainable = True
            if not has_other_trainable:
                print("  (None)")

            print("\n" + "=" * 80)
            print(f"TOTAL TRAINABLE PARAMETERS: {total_trainable:,}")
            print("=" * 80 + "\n")

        # Disable grad for frozen components in loss_module
        # These are not trainable and should not be tracked by DDP
        no_grad(self.loss_module.perceptual_loss)
        if self.loss_module.teacher_vision_model is not None:
            no_grad(self.loss_module.teacher_vision_model)
        if self.loss_module.teacher_mlp1 is not None:
            no_grad(self.loss_module.teacher_mlp1)

        # Compile models for efficiency
        # self.vae_model = torch.compile(self.vae_model)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Optimizer for encoder (generator)
        params_encoder = filter_nograd_tensors(self.vae_model.parameters())
        optimizer_encoder = self.optimizer([{"params": params_encoder}])
        lr_scheduler_encoder = get_constant_schedule_with_warmup(
            optimizer_encoder, num_warmup_steps=1000
        )

        # Optimizer for discriminator
        discriminator = self.loss_module.discriminator
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
        self._logged_images_count = 0

    def on_predict_start(self) -> None:
        self._logged_images_count = 0

    def training_step(self, batch, batch_idx):
        img, _, metadata = batch

        # Get optimizers
        opt_generator, opt_discriminator = self.optimizers()

        # Add global step to metadata
        metadata = metadata or {}
        metadata['global_step'] = self.global_step

        # Check if discriminator should be trained (for weight update decision)
        train_discriminator = self.loss_module.should_discriminator_be_trained(
            self.global_step
        )

        # Forward pass through VAE model to get reconstructions and features
        reconstructed_pixels, student_features = self.vae_model(
            img, return_features=True
        )

        # Pass student features to loss module for distillation
        extra_result_dict = {
            "student_features": student_features,
        }

        ######################
        # Optimize Generator #
        ######################
        self.toggle_optimizer(opt_generator)

        # Compute generator loss (reconstruction + perceptual + GAN)
        total_loss, loss_dict = self.loss_module(
            inputs=img,
            reconstructions=reconstructed_pixels,
            extra_result_dict=extra_result_dict,
            global_step=self.global_step,
            mode="generator",
        )

        # Backward and optimize generator (encoder)
        self.manual_backward(total_loss)
        self.clip_gradients(
            opt_generator, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        opt_generator.step()
        opt_generator.zero_grad()
        self.untoggle_optimizer(opt_generator)

        # Prepare output dict
        output_dict = {"loss": total_loss}
        output_dict.update(loss_dict)

        ##########################
        # Optimize Discriminator #
        ##########################
        if train_discriminator:
            self.toggle_optimizer(opt_discriminator)

            # Compute discriminator loss with detached reconstructions
            discriminator_loss, disc_loss_dict = self.loss_module(
                inputs=img,
                reconstructions=reconstructed_pixels.detach(),  # Detach to avoid gradient flow to generator
                extra_result_dict={},
                global_step=self.global_step,
                mode="discriminator",
            )

            # Backward and optimize discriminator
            self.manual_backward(discriminator_loss)
            self.clip_gradients(
                opt_discriminator, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )
            opt_discriminator.step()
            opt_discriminator.zero_grad()
            self.untoggle_optimizer(opt_discriminator)

            output_dict.update(disc_loss_dict)

        # Update learning rates
        sch_encoder, sch_discriminator = self.lr_schedulers()
        sch_encoder.step()
        if train_discriminator:
            sch_discriminator.step()

        # Log learning rates
        output_dict["lr_encoder"] = opt_generator.param_groups[0]['lr']
        output_dict["lr_discriminator"] = opt_discriminator.param_groups[0]['lr']

        # Log all losses
        self.log_dict(output_dict, prog_bar=True, on_step=True, sync_dist=False)

        return output_dict["loss"]

    def predict_step(self, batch, batch_idx):
        img, _, metadata = batch

        with torch.no_grad():
            # Encode to latent and decode to reconstruct
            samples = self.vae_model(img)

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
        self.loss_module.state_dict(
            destination=destination,
            prefix=prefix + "loss_module.",
            keep_vars=keep_vars,
        )
        return destination
