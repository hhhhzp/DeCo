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
from src.models.transformer.modeling_internvl_chat import InternVLChatModel
from src.utils.model_loader import ModelLoader
from src.callbacks.simple_ema import SimpleEMA
from src.diffusion.base.sampling import BaseSampler
from src.diffusion.base.training import BaseTrainer
from src.utils.no_grad import no_grad, filter_nograd_tensors
from src.utils.copy import copy_params

EMACallable = Callable[[nn.Module, nn.Module], SimpleEMA]
OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        vae: BaseAE,
        denoiser: nn.Module,
        diffusion_trainer: BaseTrainer,
        diffusion_sampler: BaseSampler,
        ema_tracker: SimpleEMA = None,
        optimizer: OptimizerCallable = None,
        lr_scheduler: LRSchedulerCallable = None,
        eval_original_model: bool = False,
        distill: bool = False,
        pretrain_model_path: str = None,
        diffusion_batch_mul: int = 4,  # Configurable latent replication factor
    ):
        super().__init__()
        self.vae = vae
        self.denoiser = denoiser
        self.ema_denoiser = copy.deepcopy(self.denoiser)
        self.diffusion_sampler = diffusion_sampler
        self.diffusion_trainer = diffusion_trainer
        self.ema_tracker = ema_tracker
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.eval_original_model = eval_original_model
        self.distill = distill
        self.diffusion_batch_mul = diffusion_batch_mul  # Multiply latent samples to improve training efficiency

        # Initialize teacher model for distillation
        if self.distill:
            self.teacher_denoiser = copy.deepcopy(self.denoiser)

        self._strict_loading = False
        self._logged_images_count = 0
        # Track how many images have been logged for comparison
        self.pretrain_model_path = pretrain_model_path

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()
        # 然后初始化vision model（如果预训练权重中不包含vision model部分）
        self.init_vision_model()

        # Initialize teacher model if distillation is enabled
        if self.distill:
            self.init_teacher_model()

        # 首先加载用户指定的预训练权重
        if self.pretrain_model_path is not None:
            checkpoint = torch.load(self.pretrain_model_path, map_location='cpu')
            msg = self.load_state_dict(checkpoint['state_dict'], strict=False)
            if self.global_rank == 0:
                print(f"Loaded pretrained model from {self.pretrain_model_path}: {msg}")

        copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)

        # disable grad for vae
        no_grad(self.vae)
        # no_grad(self.diffusion_sampler)
        no_grad(self.ema_denoiser)

        if self.distill:
            self.teacher_denoiser = torch.compile(self.teacher_denoiser)
        self.denoiser = torch.compile(self.denoiser)
        self.ema_denoiser = torch.compile(self.ema_denoiser)

    def init_vision_model(
        self,
        pretrained_model_path: str = "/apdcephfs/share_300000800/datamultimodal/models/InternVL3-2B",
    ):
        """
        从预训练的 InternVLChatModel 中加载 vision model 和 mlp1

        Args:
            pretrained_model_path: 预训练模型路径
        """
        # 从预训练模型加载 InternVLChatModel
        config = AutoConfig.from_pretrained(
            pretrained_model_path, trust_remote_code=True
        )
        config.vision_config.drop_path_rate = 0.0
        model = AutoModel.from_pretrained(
            pretrained_model_path,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # 提取 vision_model 和 mlp1
        self.denoiser.vision_model.load_state_dict(model.vision_model.state_dict())
        self.denoiser.mlp1.load_state_dict(model.mlp1.state_dict())

        # 如果不进行蒸馏，则冻结 vision_model 和 mlp1
        if not self.distill:
            no_grad(self.denoiser.vision_model)
            no_grad(self.denoiser.mlp1)

    def init_teacher_model(
        self,
        pretrained_model_path: str = "/apdcephfs/share_300000800/datamultimodal/models/InternVL3-2B",
    ):
        """
        初始化冻结的 teacher model 用于自蒸馏

        Args:
            pretrained_model_path: 预训练模型路径
        """
        # 从预训练模型加载 InternVLChatModel
        config = AutoConfig.from_pretrained(
            pretrained_model_path, trust_remote_code=True
        )
        config.vision_config.drop_path_rate = 0.0
        model = AutoModel.from_pretrained(
            pretrained_model_path,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # 提取 vision_model 和 mlp1 到 teacher model
        self.teacher_denoiser.vision_model.load_state_dict(
            model.vision_model.state_dict()
        )
        self.teacher_denoiser.mlp1.load_state_dict(model.mlp1.state_dict())

        # 冻结整个 teacher model
        no_grad(self.teacher_denoiser)

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return [self.ema_tracker]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        params_denoiser = filter_nograd_tensors(self.denoiser.parameters())
        params_trainer = filter_nograd_tensors(self.diffusion_trainer.parameters())
        params_sampler = filter_nograd_tensors(self.diffusion_sampler.parameters())
        param_groups = [
            {
                "params": params_denoiser,
            },
            {
                "params": params_trainer,
            },
            {"params": params_sampler, "lr": 1e-3},
        ]
        # optimizer: torch.optim.Optimizer = self.optimizer([*params_trainer, *params_denoiser])
        optimizer: torch.optim.Optimizer = self.optimizer(param_groups)
        # lr_scheduler = get_constant_schedule_with_warmup(
        #     optimizer, num_warmup_steps=1000
        # )
        return dict(
            optimizer=optimizer,
            # lr_scheduler={
            #     "scheduler": lr_scheduler,
            #     "interval": "step",
            #     "frequency": 1,
            # },
        )

    def on_validation_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self._logged_images_count = 0  # Reset counter at validation start

    def on_predict_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self._logged_images_count = 0  # Reset counter at predict start

    # sanity check before training start
    def on_train_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self.ema_tracker.setup_models(net=self.denoiser, ema_net=self.ema_denoiser)

    def training_step(self, batch, batch_idx):
        # For reconstruction task: input image is both source and target
        img, _, metadata = batch  # img is the original image
        # if self.global_rank == 0:
        #     print(self.eval_original_model, "eval_original_model")
        #     print(img.shape, "img.shape")
        with torch.no_grad():
            # Encode image to latent space for diffusion
            x = self.vae.encode(img)

        # Extract condition from original image (only once per image)
        # This replaces the external text/class condition
        vit_embeds = self.denoiser.extract_vision_feature(img)
        condition = self.denoiser.forward_condition(img, vit_embeds=vit_embeds)

        # Replicate latent x to improve training efficiency
        # Since condition extraction is more expensive, we reuse the same condition
        # with multiple copies of x to increase training samples
        x = x.repeat(self.diffusion_batch_mul, 1, 1, 1)
        condition = condition.repeat(self.diffusion_batch_mul, 1, 1)

        # Run diffusion training with extracted condition
        # No uncondition needed for reconstruction task
        loss = self.diffusion_trainer(
            self.denoiser,
            self.ema_denoiser,
            self.diffusion_sampler,
            x,
            condition=condition,
            uncondition=None,  # No CFG for reconstruction
            metadata=metadata,
        )

        # Add distillation loss if enabled
        if self.distill:
            with torch.no_grad():
                # Extract teacher features using frozen teacher model
                teacher_vit_embeds = self.teacher_denoiser.extract_feature(img).detach()

            # MSE loss for feature alignment
            distill_loss = torch.nn.functional.mse_loss(vit_embeds, teacher_vit_embeds)

            # Add distillation loss to total loss
            loss["distill_loss"] = distill_loss
            loss["loss"] = loss["loss"] + distill_loss

        # Log learning rate for LR curve tracking
        if self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]['lr']
            loss["learning_rate"] = current_lr

        # to be do! fix the bug in tqdm iteration when enabling accumulate_grad_batches>1
        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)
        return loss["loss"]

    def predict_step(self, batch, batch_idx):
        # For reconstruction: input image, noise, metadata
        img, _, metadata = batch

        generator = torch.Generator().manual_seed(42)
        x_t = torch.randn(
            img.shape,
            generator=generator,
            dtype=torch.float32,
        ).to(img.device)

        # Select model and compute condition only once
        model = self.denoiser if self.eval_original_model else self.ema_denoiser

        with torch.no_grad():
            # Extract condition from input image (only once)
            condition = model.forward_condition(img)
            # sample images (no uncondition for reconstruction)
            samples = self.diffusion_sampler(
                model, x_t, condition, uncondition=condition
            )
            samples = self.vae.decode(samples)

            # Log first 6 images comparison (original vs reconstructed)
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

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle _orig_mod prefix compatibility"""
        # Clean the incoming state_dict by removing _orig_mod prefix
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.replace('_orig_mod.', '')
            cleaned_state_dict[clean_key] = value

        # Use the cleaned state_dict for loading
        return super().load_state_dict(cleaned_state_dict, strict=strict)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)
        self.denoiser.state_dict(
            destination=destination, prefix=prefix + "denoiser.", keep_vars=keep_vars
        )
        self.ema_denoiser.state_dict(
            destination=destination,
            prefix=prefix + "ema_denoiser.",
            keep_vars=keep_vars,
        )
        self.diffusion_trainer.state_dict(
            destination=destination,
            prefix=prefix + "diffusion_trainer.",
            keep_vars=keep_vars,
        )
        return destination
