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
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from src.models.autoencoder.base import fp2uint8
from src.models.uniflow.modeling_uniflow import UniFlowVisionModel
from src.models.uniflow.modeling_uniflow_dcae import UniFlowVisionModel_DCAE
from src.models.uniflow.configuration_uniflow import UniFlowVisionConfig
from src.callbacks.simple_ema import SimpleEMA
from src.utils.no_grad import no_grad, filter_nograd_tensors
from src.utils.copy import copy_params
from transformers import (
    AutoModel,
    AutoConfig,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

# Log to wandb
import wandb
import numpy as np


OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]


class LightningUniFlowModel(pl.LightningModule):
    """
    Lightning wrapper for UniFlowVisionModel.
    UniFlow is an end-to-end model that directly reconstructs images from pixel space.
    """

    def __init__(
        self,
        config_path: str = None,
        ema_tracker: SimpleEMA = None,
        optimizer: OptimizerCallable = None,
        lr_scheduler: LRSchedulerCallable = None,
        eval_original_model: bool = False,
        pretrain_model_path: str = None,
        use_ema: bool = True,
        distill: bool = False,
        train_semantic_ae: bool = False,
        frozen_encoder: bool = True,
        frozen_mlp: bool = True,
        resume: bool = False,
    ):
        super().__init__()
        config = UniFlowVisionConfig.from_pretrained(config_path)
        self.model = UniFlowVisionModel(config)
        self.use_ema = use_ema
        self.train_semantic_ae = train_semantic_ae
        self.frozen_encoder = frozen_encoder
        self.frozen_mlp = frozen_mlp
        self.resume = resume
        self.distill = distill

        # Create EMA model if enabled
        if self.use_ema:
            self.ema_model = copy.deepcopy(self.model)
            no_grad(self.ema_model)
        else:
            self.ema_model = None

        # Create teacher model if distillation is enabled
        if self.distill:
            self.teacher_model = None  # Will be initialized in configure_model
        else:
            self.teacher_model = None

        self.ema_tracker = ema_tracker
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.eval_original_model = eval_original_model
        self.pretrain_model_path = pretrain_model_path

        self._strict_loading = False
        self._logged_images_count = 0

    def print_trainable_parameters(self):
        """
        Print trainable parameters of self.model (only on rank 0)
        """
        if self.global_rank != 0:
            return

        print("\n" + "=" * 80)
        print("Trainable Parameters in self.model:")
        print("=" * 80)

        trainable_params = []
        frozen_params = []
        total_trainable = 0
        total_frozen = 0

        for name, param in self.model.named_parameters():
            num_params = param.numel()
            if param.requires_grad:
                trainable_params.append((name, num_params))
                total_trainable += num_params
            else:
                frozen_params.append((name, num_params))
                total_frozen += num_params

        # Print trainable parameters
        print(f"\n✓ TRAINABLE MODULES ({len(trainable_params)} parameters):")
        for name, num_params in trainable_params:
            print(f"  - {name}: {num_params:,} params")

        # Print frozen parameters
        print(f"\n✗ FROZEN MODULES ({len(frozen_params)} parameters):")
        for name, num_params in frozen_params:
            print(f"  - {name}: {num_params:,} params")

        # Print summary
        total_params = total_trainable + total_frozen
        trainable_percent = (
            100 * total_trainable / total_params if total_params > 0 else 0
        )
        print("\n" + "-" * 80)
        print(f"SUMMARY:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {total_trainable:,} ({trainable_percent:.2f}%)")
        print(f"  Frozen parameters: {total_frozen:,} ({100-trainable_percent:.2f}%)")
        print("=" * 80 + "\n")

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
        msg = self.model.load_state_dict(model.vision_model.state_dict(), strict=False)
        if self.global_rank == 0:
            print(f"Loaded vision_model and mlp1 from {pretrained_model_path}: {msg}")
        self.model.mlp1.load_state_dict(model.mlp1.state_dict())

        if hasattr(self.model, "shallow_encoder"):
            msg = self.model.shallow_encoder.load_state_dict(
                model.vision_model.encoder.state_dict(), strict=False
            )
            print(f"Loaded shallow_encoder from {pretrained_model_path}: {msg}")
            self.model.shallow_embeddings.load_state_dict(
                model.vision_model.embeddings.state_dict()
            )

    def init_teacher_model(
        self,
        pretrained_model_path: str = "/apdcephfs/share_300000800/datamultimodal/models/InternVL3-2B",
    ):
        """
        Initialize lightweight frozen teacher model for distillation.
        Only loads vision_model and mlp1 components needed for extract_feature.

        Args:
            pretrained_model_path: 预训练模型路径
        """
        # 从预训练模型加载完整模型
        config = AutoConfig.from_pretrained(
            pretrained_model_path, trust_remote_code=True
        )
        config.vision_config.drop_path_rate = 0.0
        full_model = AutoModel.from_pretrained(
            pretrained_model_path,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # 创建轻量级 teacher model
        self.teacher_model = LightweightTeacherModel(
            vision_model=full_model.vision_model,
            mlp1=full_model.mlp1,
            select_layer=full_model.select_layer,
            downsample_ratio=full_model.downsample_ratio,
            ps_version=full_model.ps_version,
        )

        # 删除完整模型以释放内存
        del full_model

        if self.global_rank == 0:
            print(
                f"Initialized lightweight frozen teacher model from {pretrained_model_path}"
            )
            print(
                "Teacher model only contains: vision_model, mlp1, and extract_feature method"
            )

    def configure_model(self) -> None:
        """Initialize model weights and load pretrained checkpoints"""
        self.trainer.strategy.barrier()

        # Load pretrained weights if specified
        if self.pretrain_model_path is not None:
            checkpoint = torch.load(self.pretrain_model_path, map_location='cpu')
            msg = self.load_state_dict(checkpoint['state_dict'], strict=False)
            if self.global_rank == 0:
                print(f"Loaded pretrained model from {self.pretrain_model_path}: {msg}")
        else:
            self.init_vision_model()
        if self.distill:
            self.init_teacher_model()

        # Copy parameters to EMA model after loading checkpoint if not resuming
        if not self.resume and self.use_ema:
            copy_params(src_model=self.model, dst_model=self.ema_model)
            if self.global_rank == 0:
                print("Copied parameters from model to EMA model")

        if self.teacher_model is not None:
            no_grad(self.teacher_model)
            if self.global_rank == 0:
                print("Frozen teacher model for distillation")

        # Freeze encoder components if specified
        if self.frozen_encoder:
            no_grad(self.model.embeddings)
            no_grad(self.model.encoder)
            if self.global_rank == 0:
                print("Frozen encoder (embeddings and encoder)")

        # Freeze mlp if specified
        if self.frozen_mlp:
            no_grad(self.model.mlp1)
            if self.global_rank == 0:
                print("Frozen mlp1")

        # Print trainable parameters summary (only on rank 0)
        self.print_trainable_parameters()

        # self.model = torch.compile(self.model)

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        """Configure EMA callback"""
        if self.ema_tracker is not None:
            return [self.ema_tracker]
        return []

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and learning rate scheduler with different learning rates"""
        # Define prefixes for vision encoder components (embeddings, encoder, mlp1)
        vision_encoder_prefixes = (
            'embeddings.',
            'encoder.',
            'mlp1.',
            '_orig_mod.embeddings.',
            '_orig_mod.encoder.',
            '_orig_mod.mlp1.',
        )

        # Separate vision encoder parameters
        vision_encoder_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(vision_encoder_prefixes):
                vision_encoder_params.append(param)
            else:
                other_params.append(param)

        # Build parameter groups with custom names
        param_groups = [
            {"params": other_params, "name": "default"},  # Default learning rate
            {
                "params": vision_encoder_params,
                "lr": 1e-5,
                "name": "vision_encoder",
            },  # Lower learning rate for vision encoder
        ]

        optimizer: torch.optim.Optimizer = self.optimizer(param_groups)

        if self.distill:
            lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer,
                num_warmup_steps=10000,
                num_training_steps=200000,
                min_lr=1e-5,
            )
            return dict(
                optimizer=optimizer,
                lr_scheduler={
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            )
        return dict(optimizer=optimizer)

    def on_validation_start(self) -> None:
        """Prepare for validation"""
        self._logged_images_count = 0

    def on_predict_start(self) -> None:
        """Prepare for prediction"""
        self._logged_images_count = 0

    def on_train_start(self) -> None:
        """Setup EMA tracking before training"""
        if self.use_ema and self.ema_model is not None:
            self.ema_model.to(torch.float32)
            if self.ema_tracker is not None:
                self.ema_tracker.setup_models(net=self.model, ema_net=self.ema_model)

    def training_step(self, batch, batch_idx):
        """
        Training step for UniFlow model.
        Supports two modes:
        1. Flow matching training (default)
        2. Semantic autoencoder training (train_semantic_ae=True)
        """
        # Unpack batch: input image is both source and target for reconstruction
        img, _, metadata = batch
        if self.distill and self.teacher_model is not None:
            with torch.no_grad():
                teacher_img = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(
                    img * 0.5 + 0.5
                )
                teacher_feat = self.teacher_model.extract_feature(teacher_img)
        else:
            teacher_feat = None
        loss_dict = self.model.forward_loss(img, teacher_feat=teacher_feat)

        # Compute total loss
        total_loss = loss_dict["loss"]

        # Log learning rate for different parameter groups
        if self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            for group_idx, param_group in enumerate(optimizer.param_groups):
                # Use custom name if available, otherwise use group index
                group_name = param_group.get('name', f'group_{group_idx}')
                lr_key = f"lr/{group_name}"
                loss_dict[lr_key] = param_group['lr']

        # Log metrics
        self.log_dict(loss_dict, prog_bar=True, on_step=True, sync_dist=False)
        return total_loss

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for UniFlow model.
        Reconstructs images and logs comparison visualizations.
        """
        img, _, metadata = batch

        # Select model (original or EMA)
        model = (
            self.model
            if self.eval_original_model
            else (self.ema_model if self.use_ema else self.model)
        )

        with torch.no_grad():
            # Apply padding if dimensions are not divisible by 28
            _, _, h, w = img.shape
            pad_h = (28 - h % 28) % 28
            pad_w = (28 - w % 28) % 28

            if pad_h > 0 or pad_w > 0:
                # Apply uniform padding on all sides
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                img_padded = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant',
                    value=0,
                )

                # Forward pass to reconstruct images
                samples = model(img_padded)

                # Remove padding from output
                samples = samples[:, :, pad_top : pad_top + h, pad_left : pad_left + w]
            else:
                # Forward pass to reconstruct images
                samples = model(img)

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

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Custom state_dict to save model, mlp1 and EMA model separately"""
        if destination is None:
            destination = {}
        self._save_to_state_dict(destination, prefix, keep_vars)

        # Save main model (UniFlowVisionModel)
        model_state = self.model.state_dict(keep_vars=keep_vars)
        for key, value in model_state.items():
            destination[prefix + "model." + key] = value

        if hasattr(self, 'mlp1'):
            mlp1_state = self.mlp1.state_dict(keep_vars=keep_vars)
            for key, value in mlp1_state.items():
                destination[prefix + "mlp1." + key] = value

        # Save EMA model if enabled
        if self.use_ema and self.ema_model is not None:
            ema_state = self.ema_model.state_dict(keep_vars=keep_vars)
            for key, value in ema_state.items():
                destination[prefix + "ema_model." + key] = value

        return destination

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle DDP and torch.compile prefixes.
        Cleans up 'module._orig_mod.' patterns from checkpoint keys.
        """
        # Clean up DDP and torch.compile prefixes
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # Remove module and _orig_mod prefixes
            new_key = new_key.replace('.module.', '.')
            new_key = new_key.replace('._orig_mod.', '.')
            new_state_dict[new_key] = value

        # Call parent's load_state_dict with cleaned keys
        return super().load_state_dict(new_state_dict, strict=strict)


# 创建轻量级 teacher model，只包含必要组件
class LightweightTeacherModel(nn.Module):
    def __init__(self, vision_model, mlp1, select_layer, downsample_ratio, ps_version):
        super().__init__()
        self.vision_model = vision_model
        self.mlp1 = mlp1
        self.select_layer = select_layer
        self.downsample_ratio = downsample_ratio
        self.ps_version = ps_version

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds_mlp = self.mlp1(vit_embeds)
        return {"vit_embeds": vit_embeds, "vit_embeds_mlp": vit_embeds_mlp}
