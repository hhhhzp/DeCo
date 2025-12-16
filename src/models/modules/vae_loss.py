"""
Simplified loss module for VAE training without VQ quantizer.
"""

from typing import Mapping, Text, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.models.modules.perceptual_loss import PerceptualLoss
from src.models.modules.discriminator import NLayerDiscriminator
from src.models.transformer.configuration_internvl_chat import InternVLChatConfig
from src.models.transformer.modeling_intern_vit import InternVisionModel
from src.utils.no_grad import no_grad


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator."""
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor,
) -> torch.Tensor:
    """Computes the LeCam loss for regularization."""
    lecam_loss = torch.mean(
        torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2)
    )
    lecam_loss += torch.mean(
        torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2)
    )
    return lecam_loss


class VAEReconstructionLoss(nn.Module):
    """
    Reconstruction loss for VAE training.
    Includes: MSE/L1 loss, LPIPS perceptual loss, and GAN loss.
    """

    def __init__(
        self,
        discriminator_start: int = 500000,
        discriminator_factor: float = 1.0,
        discriminator_weight: float = 0.1,
        perceptual_loss: str = "lpips-convnext_s-1.0-0.1",
        perceptual_weight: float = 1.1,
        reconstruction_loss: str = "l2",
        reconstruction_weight: float = 1.0,
        lecam_regularization_weight: float = 0.001,
        kl_weight: float = 0.0,
        logvar_init: float = 0.0,
        distillation_weight: float = 1.0,
        distillation_loss_type: str = "mse",
        teacher_model_path: str = None,
        select_layer: int = -1,
        downsample_ratio: float = 0.5,
    ):
        """
        Args:
            discriminator_start: Step to start training discriminator
            discriminator_factor: Factor for discriminator loss
            discriminator_weight: Weight for discriminator loss
            perceptual_loss: Perceptual loss configuration
            perceptual_weight: Weight for perceptual loss
            reconstruction_loss: Type of reconstruction loss ("l1" or "l2")
            reconstruction_weight: Weight for reconstruction loss
            lecam_regularization_weight: Weight for LeCam regularization
            kl_weight: Weight for KL divergence (if using VAE)
            logvar_init: Initial value for log variance
            distillation_weight: Weight for distillation loss
            distillation_loss_type: Type of distillation loss ("mse" or "cosine")
            teacher_model_path: Path to pretrained teacher model (InternVL)
            select_layer: Which layer to use from vision model (-1 for last)
            downsample_ratio: Downsample ratio for pixel shuffle
        """
        super().__init__()

        # Teacher model parameters
        self.teacher_model_path = teacher_model_path
        self.select_layer = select_layer
        self.downsample_ratio = downsample_ratio
        self.teacher_vision_model = None
        self.teacher_mlp1 = None

        # Initialize teacher model if path is provided
        if teacher_model_path is not None and distillation_weight > 0.0:
            self.init_teacher_model(teacher_model_path)

        self.discriminator = NLayerDiscriminator()
        self.perceptual_loss = PerceptualLoss(perceptual_loss).eval()
        no_grad(self.perceptual_loss)
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight

        self.discriminator_iter_start = discriminator_start
        self.discriminator_factor = discriminator_factor
        self.discriminator_weight = discriminator_weight

        self.lecam_regularization_weight = lecam_regularization_weight
        self.lecam_ema_decay = 0.999

        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.kl_weight = kl_weight
        if self.kl_weight > 0.0:
            self.logvar = nn.Parameter(
                torch.ones(size=()) * logvar_init, requires_grad=False
            )

        self.distillation_weight = distillation_weight
        self.distillation_loss_type = distillation_loss_type

    def init_teacher_model(self, pretrained_model_path: str):
        """
        Initialize frozen teacher model for self-distillation.

        Args:
            pretrained_model_path: Path to pretrained model
        """
        print(f"Loading teacher model from {pretrained_model_path}...")

        # Load pretrained InternVLChatModel config
        config = InternVLChatConfig.from_pretrained(pretrained_model_path)
        vision_config = config.vision_config
        vision_config.drop_path_rate = 0.0

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        # Create teacher vision model
        self.teacher_vision_model = InternVisionModel(vision_config)

        # Create teacher mlp1
        self.teacher_mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2,
                llm_hidden_size,
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        # Load pretrained weights
        model = AutoModel.from_pretrained(
            pretrained_model_path,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Extract vision_model and mlp1 to teacher model
        self.teacher_vision_model.load_state_dict(model.vision_model.state_dict())
        self.teacher_mlp1.load_state_dict(model.mlp1.state_dict())

        # Freeze entire teacher model
        for param in self.teacher_vision_model.parameters():
            param.requires_grad = False
        for param in self.teacher_mlp1.parameters():
            param.requires_grad = False

        # Set to eval mode
        self.teacher_vision_model.eval()
        self.teacher_mlp1.eval()

        print("Teacher model loaded and frozen successfully!")
        no_grad(self.teacher_vision_model)
        no_grad(self.teacher_mlp1)

    def pixel_shuffle(self, x, scale_factor=0.5):
        """Pixel shuffle downsampling"""
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_teacher_features(self, pixel_values):
        """
        Extract features from frozen teacher model.
        :param pixel_values: input image [B, C, H, W] in range [-1, 1]
        :return: teacher features [B, num_patches, hidden_size]
        """
        if self.teacher_vision_model is None or self.teacher_mlp1 is None:
            return None

        with torch.no_grad():
            # Normalize to ImageNet stats
            pixel_values = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(
                pixel_values * 0.5 + 0.5
            )

            # Extract vision features
            if self.select_layer == -1:
                vit_embeds = self.teacher_vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=False,
                    return_dict=True,
                ).last_hidden_state
            else:
                vit_embeds = self.teacher_vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states[self.select_layer]

            vit_embeds = vit_embeds[:, 1:, :]  # Remove CLS token

            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.pixel_shuffle(
                vit_embeds, scale_factor=self.downsample_ratio
            )
            vit_embeds = vit_embeds.reshape(
                vit_embeds.shape[0], -1, vit_embeds.shape[-1]
            )
            vit_embeds = self.teacher_mlp1(vit_embeds)

        return vit_embeds

    def should_discriminator_be_trained(self, global_step: int):
        """Check if discriminator should be trained at this step."""
        return global_step >= self.discriminator_iter_start

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
        mode: str = "generator",
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """
        Forward pass for loss computation.

        Args:
            inputs: Original images [B, C, H, W] in range [-1, 1]
            reconstructions: Reconstructed images [B, C, H, W]
            extra_result_dict: Extra results (e.g., KL divergence)
            global_step: Current training step
            mode: "generator" or "discriminator"

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(
                inputs, reconstructions, extra_result_dict, global_step
            )
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def _forward_generator(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        # Extract student features from extra_result_dict
        student_features = extra_result_dict.get("student_features", None)

        # Extract teacher features using teacher model
        teacher_features = None
        if self.distillation_weight > 0.0 and self.teacher_vision_model is not None:
            teacher_features = self.extract_teacher_features(inputs)

        # Convert from [-1, 1] to [0, 1] for loss computation
        inputs_01 = (inputs + 1) / 2
        reconstructions_01 = (reconstructions + 1) / 2

        # Compute reconstruction loss (MSE or L1)
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(
                inputs_01, reconstructions_01, reduction="mean"
            )
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(
                inputs_01, reconstructions_01, reduction="mean"
            )
        else:
            raise ValueError(
                f"Unsupported reconstruction_loss {self.reconstruction_loss}"
            )

        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss (LPIPS)
        perceptual_loss = self.perceptual_loss(inputs_01, reconstructions_01).mean()

        # Compute GAN loss
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = (
            self.discriminator_factor
            if self.should_discriminator_be_trained(global_step)
            else 0
        )
        d_weight = 1.0

        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions_01)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # Compute distillation loss
        distillation_loss = torch.zeros((), device=inputs.device)
        if (
            self.distillation_weight > 0.0
            and student_features is not None
            and teacher_features is not None
        ):
            if self.distillation_loss_type == "mse":
                distillation_loss = F.mse_loss(
                    student_features, teacher_features, reduction="mean"
                )
            elif self.distillation_loss_type == "cosine":
                # Cosine similarity loss (1 - cosine_similarity)
                student_norm = F.normalize(student_features, p=2, dim=-1)
                teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
                cosine_sim = (student_norm * teacher_norm).sum(dim=-1).mean()
                distillation_loss = 1.0 - cosine_sim
            else:
                raise ValueError(
                    f"Unsupported distillation_loss_type {self.distillation_loss_type}"
                )

            distillation_loss *= self.distillation_weight

        # Compute total loss
        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + d_weight * discriminator_factor * generator_loss
            + distillation_loss
        )

        # Build loss dictionary
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            weighted_gan_loss=(
                d_weight * discriminator_factor * generator_loss
            ).detach(),
            discriminator_factor=torch.tensor(discriminator_factor),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
            distillation_loss=distillation_loss.detach(),
        )

        return total_loss, loss_dict

    def _forward_discriminator(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        global_step: int,
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discriminator training step."""
        discriminator_factor = (
            self.discriminator_factor
            if self.should_discriminator_be_trained(global_step)
            else 0
        )

        # Convert from [-1, 1] to [0, 1]
        inputs_01 = (inputs + 1) / 2
        reconstructions_01 = (reconstructions + 1) / 2

        # Enable discriminator gradients
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs_01.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions_01.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(
            logits_real=logits_real, logits_fake=logits_fake
        )

        # Optional LeCam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = (
                compute_lecam_loss(
                    torch.mean(logits_real),
                    torch.mean(logits_fake),
                    self.ema_real_logits_mean,
                    self.ema_fake_logits_mean,
                )
                * self.lecam_regularization_weight
            )

            self.ema_real_logits_mean = (
                self.ema_real_logits_mean * self.lecam_ema_decay
                + torch.mean(logits_real).detach() * (1 - self.lecam_ema_decay)
            )
            self.ema_fake_logits_mean = (
                self.ema_fake_logits_mean * self.lecam_ema_decay
                + torch.mean(logits_fake).detach() * (1 - self.lecam_ema_decay)
            )

        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )

        return discriminator_loss, loss_dict
