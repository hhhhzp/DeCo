from typing import Callable, Iterable, Any, Optional, Union, Sequence, Mapping, Dict
import os.path
import copy
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from lightning.pytorch.callbacks import Callback
from transformers import AutoModel

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

        self._strict_loading = False

    def configure_model(self) -> None:
        self.trainer.strategy.barrier()
        copy_params(src_model=self.denoiser, dst_model=self.ema_denoiser)

        # disable grad for vae
        no_grad(self.vae)
        # no_grad(self.diffusion_sampler)
        no_grad(self.ema_denoiser)
        self.init_vision_model()
        # torch.compile
        self.denoiser.compile()
        self.ema_denoiser.compile()

    def init_vision_model(
        self,
        pretrained_model_path: str = "/apdcephfs/share_300000800/datamultimodal/models/InternVL3-1B",
    ):
        """
        从预训练的 InternVLChatModel 中加载冻结的 teacher model 用于自蒸馏

        Args:
            pretrained_model_path: 预训练模型路径
            force_image_size: 如果指定，会 resize position embeddings 到该尺寸
        """
        # 从预训练模型加载 InternVLChatModel
        model = AutoModel.from_pretrained(
            pretrained_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # 提取 vision_model 和 mlp1
        self.denoiser.vision_model.load_state_dict(model.vision_model.state_dict())
        self.denoiser.mlp1.load_state_dict(model.mlp1.state_dict())

        no_grad(self.denoiser.vision_model)
        no_grad(self.denoiser.mlp1)

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
        if self.lr_scheduler is None:
            return dict(optimizer=optimizer)
        else:
            lr_scheduler = self.lr_scheduler(optimizer)
            return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)

    def on_validation_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    def on_predict_start(self) -> None:
        self.ema_denoiser.to(torch.float32)

    # sanity check before training start
    def on_train_start(self) -> None:
        self.ema_denoiser.to(torch.float32)
        self.ema_tracker.setup_models(net=self.denoiser, ema_net=self.ema_denoiser)

    def training_step(self, batch, batch_idx):
        no_grad(self.denoiser.vision_model)
        no_grad(self.denoiser.mlp1)
        # For reconstruction task: input image is both source and target
        img, _, metadata = batch  # img is the original image

        with torch.no_grad():
            # Encode image to latent space for diffusion
            x = self.vae.encode(img)

        # Extract condition from original image (only once per image)
        # This replaces the external text/class condition
        # condition = self.denoiser.forward_condition(img)

        # Run diffusion training with extracted condition
        # No uncondition needed for reconstruction task
        loss = self.diffusion_trainer(
            self.denoiser,
            self.ema_denoiser,
            self.diffusion_sampler,
            x,
            condition=None,
            uncondition=None,  # No CFG for reconstruction
            metadata=metadata,
        )

        # to be do! fix the bug in tqdm iteration when enabling accumulate_grad_batches>1
        for k, v in loss.items():
            self.log(f"train/{k}", v)
        self.log_dict(loss, prog_bar=True, on_step=True, sync_dist=False)
        return loss["loss"]

    def predict_step(self, batch, batch_idx):
        # For reconstruction: input image, noise, metadata
        img, xT, metadata = batch

        with torch.no_grad():
            # Extract condition from input image (only once)
            if self.eval_original_model:
                condition = self.denoiser.forward_condition(img)
            else:
                condition = self.ema_denoiser.forward_condition(img)

        # sample images (no uncondition for reconstruction)
        if self.eval_original_model:
            samples = self.diffusion_sampler(
                self.denoiser, xT, condition, uncondition=None
            )
        else:
            samples = self.diffusion_sampler(
                self.ema_denoiser, xT, condition, uncondition=None
            )

        samples = self.vae.decode(samples)
        # fp32 -1,1 -> uint8 0,255
        samples = fp2uint8(samples)
        return samples

    def validation_step(self, batch, batch_idx):
        samples = self.predict_step(batch, batch_idx)
        return samples

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
