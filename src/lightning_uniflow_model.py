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
from src.models.uniflow.modeling_uniflow import ChannelProjector, UniFlowVisionModel
from src.models.uniflow.configuration_uniflow import UniFlowVisionConfig
from src.callbacks.simple_ema import SimpleEMA
from src.utils.no_grad import no_grad, filter_nograd_tensors
from src.utils.copy import copy_params
from transformers import (
    AutoModel,
    AutoConfig,
    get_constant_schedule_with_warmup,
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
    ):
        super().__init__()
        config = UniFlowVisionConfig.from_pretrained(config_path)
        self.model = UniFlowVisionModel(config)
        self.use_ema = use_ema
        self.distill = distill
        self.train_semantic_ae = train_semantic_ae

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
        # self.model.sem_ae.up_proj.load_state_dict(model.mlp1.state_dict())

    def init_teacher_model(
        self,
        pretrained_model_path: str = "/apdcephfs/share_300000800/datamultimodal/models/InternVL3-2B",
    ):
        """
        Initialize frozen teacher model for distillation

        Args:
            pretrained_model_path: 预训练模型路径
        """
        # 从预训练模型加载 InternVLChatModel
        config = AutoConfig.from_pretrained(
            pretrained_model_path, trust_remote_code=True
        )
        config.vision_config.drop_path_rate = 0.0
        self.teacher_model = AutoModel.from_pretrained(
            pretrained_model_path,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Freeze teacher model

        if self.global_rank == 0:
            print(f"Initialized frozen teacher model from {pretrained_model_path}")

    def configure_model(self) -> None:
        """Initialize model weights and load pretrained checkpoints"""
        self.trainer.strategy.barrier()
        self.init_vision_model()

        # Load pretrained weights if specified
        if self.pretrain_model_path is not None:
            checkpoint = torch.load(self.pretrain_model_path, map_location='cpu')
            msg = self.load_state_dict(checkpoint['state_dict'], strict=False)
            if self.global_rank == 0:
                print(f"Loaded pretrained model from {self.pretrain_model_path}: {msg}")

        # Copy parameters to EMA model
        if self.use_ema:
            copy_params(src_model=self.model, dst_model=self.ema_model)

        if self.distill:
            self.init_teacher_model()

        # Compile models for better performance
        # self.model = torch.compile(self.model)
        # if self.use_ema:
        #     self.ema_model = torch.compile(self.ema_model)

        # Freeze strategy based on training mode
        if self.train_semantic_ae:
            # Semantic AE training: freeze all parameters first, then unfreeze sem_ae
            no_grad(self.model)  # Freeze all parameters
            # Unfreeze only sem_ae
            for param in self.model.sem_ae.parameters():
                param.requires_grad = True
            # no_grad(self.model.sem_ae.up_proj)
            if self.global_rank == 0:
                print("Training mode: Semantic Autoencoder (only sem_ae trainable)")
        elif self.distill:
            # Distillation training: freeze position embeddings only
            no_grad(self.teacher_model)
            self.model.embeddings.position_embedding.requires_grad_(False)
            if self.global_rank == 0:
                print("Training mode: Distillation (position embeddings frozen)")
        else:
            # Flow matching training: freeze encoder components
            no_grad(self.model.embeddings)
            no_grad(self.model.encoder)
            no_grad(self.model.mlp1)
            if self.global_rank == 0:
                print("Training mode: Flow Matching (encoder frozen)")

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

        # Build parameter groups
        param_groups = [
            {"params": other_params},  # Default learning rate
            {
                "params": vision_encoder_params,
                "lr": 5e-5,
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
            # get_cosine_schedule_with_warmup(
            #     optimizer, num_warmup_steps=2000, num_training_steps=200000
            # )
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

        # Mode 1: Semantic autoencoder training
        if self.train_semantic_ae:
            loss_dict = self.model.forward_semantic_loss(img)
        # Mode 2: Flow matching training (default)
        else:
            # Get teacher features if distillation is enabled
            teacher_feat = None
            if self.distill and self.teacher_model is not None:
                with torch.no_grad():
                    teacher_img = Normalize(
                        IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
                    )(img * 0.5 + 0.5)
                    teacher_feat = self.teacher_model.extract_feature(teacher_img)

            # Forward pass with flow matching loss
            # The model internally:
            # 1. Encodes image to get condition tokens
            # 2. Converts target image to patch tokens
            # 3. Samples noise and timestep
            # 4. Computes flow matching loss (velocity prediction)
            loss_dict = self.model.forward_loss(img, teacher_feat=teacher_feat)

        # Compute total loss
        total_loss = loss_dict["loss"]

        # Log learning rate
        if self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]['lr']
            loss_dict["learning_rate"] = current_lr

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
