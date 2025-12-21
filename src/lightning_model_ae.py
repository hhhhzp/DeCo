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
from transformers import (
    AutoModel,
    AutoConfig,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from src.models.autoencoder.base import BaseAE, fp2uint8
from src.models.transformer.encoder_ae import VAEModel
from src.utils.model_loader import ModelLoader
from src.utils.no_grad import no_grad, filter_nograd_tensors
from src.callbacks.simple_ema import SimpleEMA
from src.utils.copy import copy_params

# Log each comparison image with wandb
import wandb
import numpy as np

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
        freeze_encoder: bool = False,
        ema_tracker: SimpleEMA = None,
        load_ema_as_main: bool = False,
    ):
        super().__init__()
        self.vae_model = vae_model
        self.ema_vae_model = copy.deepcopy(self.vae_model)
        self.loss_module = loss_module

        self.optimizer = optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.lr_scheduler = lr_scheduler
        self.ema_tracker = ema_tracker

        self.eval_original_model = eval_original_model
        self.pretrain_model_path = pretrain_model_path
        self.freeze_encoder = freeze_encoder
        self.load_ema_as_main = load_ema_as_main

        self._strict_loading = True
        self._logged_images_count = 0

        # Automatic optimization disabled for manual discriminator training
        self.automatic_optimization = False

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()

        # Load pretrained weights BEFORE torch.compile to avoid _orig_mod prefix issues
        if self.pretrain_model_path is not None:
            checkpoint = torch.load(self.pretrain_model_path, map_location='cpu')
            state_dict = checkpoint['state_dict']

            # Clean up DDP and torch.compile prefixes while preserving module structure
            # Expected format: vae_model.module._orig_mod.xxx or loss_module.module._orig_mod.xxx
            # Target format: vae_model.xxx or loss_module.xxx
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                # Remove 'module._orig_mod.' pattern (DDP + torch.compile)
                new_key = new_key.replace('.module._orig_mod.', '.')
                # Also handle cases with only one of the prefixes
                new_key = new_key.replace('.module.', '.')
                new_key = new_key.replace('._orig_mod.', '.')

                # If load_ema_as_main is True, load ema_vae_model weights into vae_model
                if self.load_ema_as_main and new_key.startswith('ema_vae_model.'):
                    new_key = new_key.replace('ema_vae_model.', 'vae_model.', 1)

                new_state_dict[new_key] = value

            msg = self.load_state_dict(new_state_dict, strict=False)
            if self.global_rank == 0:
                print(f"\nLoaded pretrained model from {self.pretrain_model_path}")
                if self.load_ema_as_main:
                    print("✓ Loaded EMA model weights as main model (vae_model)")
                print(f"Loading status: {msg}")

        # Copy parameters from vae_model to ema_vae_model
        copy_params(src_model=self.vae_model, dst_model=self.ema_vae_model)

        # Disable grad for frozen components in loss_module
        # These are not trainable and should not be tracked by DDP
        if self.freeze_encoder:
            no_grad(self.vae_model.vision_model)
            no_grad(self.vae_model.mlp1)
        no_grad(self.loss_module.perceptual_loss)
        if self.loss_module.teacher_vision_model is not None:
            no_grad(self.loss_module.teacher_vision_model)
        if self.loss_module.teacher_mlp1 is not None:
            no_grad(self.loss_module.teacher_mlp1)

        # Disable grad for EMA model
        no_grad(self.ema_vae_model)

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

        # Compile models for efficiency AFTER loading pretrained weights
        self.vae_model = torch.compile(self.vae_model)
        self.ema_vae_model = torch.compile(self.ema_vae_model)
        self.loss_module = torch.compile(self.loss_module)

    def _get_module(self, module):
        """Helper to unwrap DDP module if needed."""
        from torch.nn.parallel import DistributedDataParallel as DDP

        if isinstance(module, DDP):
            return module.module
        return module

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Optimizer for encoder (generator)
        vae_model = self._get_module(self.vae_model)

        # Separate parameters: vision_model and mlp1 use 0.1x learning rate
        vision_mlp_params = []
        other_params = []

        for name, param in vae_model.named_parameters():
            if param.requires_grad:
                if 'vision_model' in name or 'mlp1' in name:
                    vision_mlp_params.append(param)
                else:
                    other_params.append(param)

        # Create parameter groups with different learning rates
        # Note: The optimizer callable should be configured with base_lr in config
        # We'll manually set lr for vision_mlp group to be 0.1x of base_lr
        # Assuming base_lr = 1e-4, vision_mlp_lr = 1e-5
        param_groups = []
        if len(vision_mlp_params) > 0:
            # Group 0: vision_model and mlp1 with 0.1x learning rate
            param_groups.append({"params": vision_mlp_params, "lr": 1e-5})
        if len(other_params) > 0:
            # Group 1: other parameters with base learning rate (from config)
            param_groups.append({"params": other_params})

        optimizer_encoder = self.optimizer(param_groups)
        lr_scheduler_encoder = get_constant_schedule_with_warmup(
            optimizer_encoder, num_warmup_steps=0
        )

        # get_constant_schedule_with_warmup(optimizer_encoder, num_warmup_steps=0)

        # Optimizer for discriminator
        loss_module = self._get_module(self.loss_module)
        discriminator = loss_module.discriminator
        params_discriminator = filter_nograd_tensors(discriminator.parameters())
        optimizer_discriminator = self.discriminator_optimizer(
            [{"params": params_discriminator}]
        )
        lr_scheduler_discriminator = get_constant_schedule_with_warmup(
            optimizer_discriminator, num_warmup_steps=0
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

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        if self.ema_tracker is not None:
            return [self.ema_tracker]
        return []

    def on_validation_start(self) -> None:
        self.ema_vae_model.to(torch.float32)
        self._logged_images_count = 0

    def on_predict_start(self) -> None:
        self.ema_vae_model.to(torch.float32)
        self._logged_images_count = 0

    def on_train_start(self) -> None:
        self.ema_vae_model.to(torch.float32)
        if self.ema_tracker is not None:
            self.ema_tracker.setup_models(
                net=self.vae_model, ema_net=self.ema_vae_model
            )

    def training_step(self, batch, batch_idx):
        img, _, metadata = batch

        # Get optimizers and schedulers
        opt_generator, opt_discriminator = self.optimizers()
        sch_encoder, sch_discriminator = self.lr_schedulers()

        # Add global step to metadata
        metadata = metadata or {}
        metadata['global_step'] = self.global_step

        # Unwrap DDP module if needed
        loss_module = self._get_module(self.loss_module)

        # Check if discriminator should be trained (for weight update decision)
        train_discriminator = loss_module.should_discriminator_be_trained(
            self.global_step
        )

        # Forward pass: encode -> sample -> decode
        # Use stochastic sampling (use_mode=False) for training
        reconstructed_pixels, student_features = self.vae_model(
            img, return_features=True, use_mode=False
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
        total_loss, loss_dict = loss_module(
            inputs=img,
            reconstructions=reconstructed_pixels,
            extra_result_dict=extra_result_dict,
            global_step=self.global_step,
            mode="generator",
        )

        # Add KL loss for Power Spherical regularization
        # total_loss = total_loss + 0.0 * kl_loss
        # loss_dict["kl_loss"] = kl_loss

        # Backward and optimize generator (encoder)
        opt_generator.zero_grad()
        self.manual_backward(total_loss)
        self.clip_gradients(
            opt_generator, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        opt_generator.step()
        sch_encoder.step()

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
            discriminator_loss, disc_loss_dict = loss_module(
                inputs=img,
                reconstructions=reconstructed_pixels.detach(),  # Detach to avoid gradient flow to generator
                extra_result_dict={},
                global_step=self.global_step,
                mode="discriminator",
            )

            # Backward and optimize discriminator
            opt_discriminator.zero_grad()
            self.manual_backward(discriminator_loss)
            self.clip_gradients(
                opt_discriminator, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
            )
            opt_discriminator.step()
            sch_discriminator.step()

            self.untoggle_optimizer(opt_discriminator)

            output_dict.update(disc_loss_dict)

        # Log learning rates for different parameter groups
        # Group 0: vision_model and mlp1 (if exists)
        # Group 1: other parameters (if exists)
        if len(opt_generator.param_groups) > 1:
            output_dict["lr_encoder_vision_mlp"] = opt_generator.param_groups[0]['lr']
            output_dict["lr_encoder_other"] = opt_generator.param_groups[1]['lr']
        else:
            # Only one group exists
            output_dict["lr_encoder"] = opt_generator.param_groups[0]['lr']
        output_dict["lr_discriminator"] = opt_discriminator.param_groups[0]['lr']

        # Log all losses
        self.log_dict(output_dict, prog_bar=True, on_step=True, sync_dist=False)

        return output_dict["loss"]

    def predict_step(self, batch, batch_idx):
        img, _, metadata = batch

        # Select model based on eval_original_model flag
        model = self.vae_model if self.eval_original_model else self.ema_vae_model

        with torch.no_grad():
            # Encode to latent and decode to reconstruct
            # Use deterministic mode (use_mode=True) for inference
            samples = model(img, use_mode=True).float()

            # Denormalize reconstructions from ImageNet normalization to [0, 255] uint8
            # from src.utils.image_utils import denormalize_to_uint8

            # samples_uint8 = denormalize_to_uint8(samples, source_range="imagenet")

            # Log first 6 images comparison
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

        return fp2uint8(samples)

    def validation_step(self, batch, batch_idx):
        samples = self.predict_step(batch, batch_idx)
        return samples

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle DDP and torch.compile prefixes.
        Cleans up 'module._orig_mod.' patterns from checkpoint keys.
        """
        # Clean up DDP and torch.compile prefixes while preserving module structure
        # Expected format: vae_model.module._orig_mod.xxx or loss_module.module._orig_mod.xxx
        # Target format: vae_model.xxx or loss_module.xxx
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # Also handle cases with only one of the prefixes
            new_key = new_key.replace('.module.', '.')
            new_state_dict[new_key] = value

        # Call parent's load_state_dict with cleaned keys
        return super().load_state_dict(new_state_dict, strict=strict)

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
        self.loss_module.state_dict(
            destination=destination,
            prefix=prefix + "loss_module.",
            keep_vars=keep_vars,
        )
        return destination
