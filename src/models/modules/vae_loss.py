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


def rotate_image_batch(images: torch.Tensor, k: int) -> torch.Tensor:
    """
    Rotate images by k*90 degrees counterclockwise.

    Args:
        images: [B, C, H, W] tensor
        k: rotation factor (0: 0°, 1: 90°, 2: 180°, 3: 270°)

    Returns:
        Rotated images [B, C, H, W]
    """
    if k == 0:
        return images
    elif k == 1:  # 90° counterclockwise
        return torch.rot90(images, k=1, dims=[2, 3])
    elif k == 2:  # 180°
        return torch.rot90(images, k=2, dims=[2, 3])
    elif k == 3:  # 270° counterclockwise (90° clockwise)
        return torch.rot90(images, k=3, dims=[2, 3])
    else:
        raise ValueError(f"Invalid rotation factor k={k}, must be 0, 1, 2, or 3")


def rotate_features_back(
    features: torch.Tensor, k: int, h: int, w: int
) -> torch.Tensor:
    """
    Rotate features back to 0° orientation.

    Args:
        features: [B, N, C] tensor where N = h*w
        k: original rotation factor (will rotate back by -k*90°)
        h: feature map height
        w: feature map width

    Returns:
        Rotated features [B, N, C]
    """
    if k == 0:
        return features

    B, N, C = features.shape
    # Reshape to spatial format
    features_spatial = features.reshape(B, h, w, C).permute(0, 3, 1, 2)  # [B, C, h, w]

    # Rotate back (clockwise if original was counterclockwise)
    if k == 1:  # Was 90° CCW, rotate 90° CW (270° CCW)
        features_spatial = torch.rot90(features_spatial, k=3, dims=[2, 3])
    elif k == 2:  # Was 180°, rotate 180° back
        features_spatial = torch.rot90(features_spatial, k=2, dims=[2, 3])
    elif k == 3:  # Was 270° CCW, rotate 90° CCW
        features_spatial = torch.rot90(features_spatial, k=1, dims=[2, 3])

    # Reshape back to sequence format
    features_back = features_spatial.permute(0, 2, 3, 1).reshape(B, N, C)
    return features_back


def create_rotated_batch(pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Create a batch with 4 rotations (0°, 90°, 180°, 270°) for each image.

    Args:
        pixel_values: [B, C, H, W] tensor

    Returns:
        Rotated batch [B*4, C, H, W] where each group of 4 contains rotations of one image
    """
    rotations = []
    for k in range(4):
        rotations.append(rotate_image_batch(pixel_values, k))
    return torch.cat(rotations, dim=0)


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
        use_rotation_aug: bool = False,
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
            use_rotation_aug: Whether to use rotation augmentation (0°, 90°, 180°, 270°)
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
        self.use_rotation_aug = use_rotation_aug

    def init_teacher_model(self, pretrained_model_path: str):
        """
        Initialize frozen teacher model for self-distillation.

        Args:
            pretrained_model_path: Path to pretrained model
        """
        print(f"Loading teacher model from {pretrained_model_path}...")
        config = AutoConfig.from_pretrained(
            pretrained_model_path, trust_remote_code=True
        )
        config.vision_config.drop_path_rate = 0.0
        config.vision_config.attention_dropout = 0.0
        config.vision_config.dropout = 0.0
        model = AutoModel.from_pretrained(
            pretrained_model_path,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        # Extract vision_model and mlp1 to teacher model
        self.teacher_vision_model = model.vision_model
        self.teacher_mlp1 = model.mlp1

        # Freeze entire teacher model
        for param in self.teacher_vision_model.parameters():
            param.requires_grad = False
        for param in self.teacher_mlp1.parameters():
            param.requires_grad = False

        # Set to eval mode
        self.teacher_vision_model.eval()
        self.teacher_mlp1.eval()

        print("Teacher model loaded and frozen successfully!")

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

    def extract_teacher_features(self, pixel_values, use_rotation_aug=False):
        """
        Extract features from frozen teacher model.

        Args:
            pixel_values: input image [B, C, H, W] in range [-1, 1]
            use_rotation_aug: if True, apply rotation augmentation (0°, 90°, 180°, 270°)
                             and average features after rotating back

        Returns:
            teacher features [B, num_patches, hidden_size]
        """
        if self.teacher_vision_model is None or self.teacher_mlp1 is None:
            return None

        with torch.no_grad():
            if use_rotation_aug:
                # Create rotated versions: [B*4, C, H, W]
                B = pixel_values.shape[0]
                pixel_values_rotated = create_rotated_batch(pixel_values)

                # Normalize to ImageNet stats
                pixel_values_rotated = Normalize(
                    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
                )(pixel_values_rotated * 0.5 + 0.5)

                # Extract vision features for all rotations in one forward pass
                if self.select_layer == -1:
                    vit_embeds = self.teacher_vision_model(
                        pixel_values=pixel_values_rotated,
                        output_hidden_states=False,
                        return_dict=True,
                    ).last_hidden_state
                else:
                    vit_embeds = self.teacher_vision_model(
                        pixel_values=pixel_values_rotated,
                        output_hidden_states=True,
                        return_dict=True,
                    ).hidden_states[self.select_layer]

                vit_embeds = vit_embeds[:, 1:, :]  # Remove CLS token [B*4, N, C]

                # Get spatial dimensions
                h = w = int(vit_embeds.shape[1] ** 0.5)

                # Reshape all features to spatial format at once [B*4, h, w, C]
                vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)

                # Split into 4 groups and rotate back
                vit_embeds_list = torch.chunk(vit_embeds, 4, dim=0)
                rotated_back_list = []

                for k, vit_embeds_k in enumerate(vit_embeds_list):
                    if k == 0:
                        # No rotation needed for 0°
                        rotated_back_list.append(vit_embeds_k)
                    else:
                        # Rotate back: convert to [B, C, h, w] format
                        vit_spatial = vit_embeds_k.permute(0, 3, 1, 2)

                        # Rotate back (inverse rotation)
                        if k == 1:  # Was 90° CCW, rotate 270° CCW (90° CW)
                            vit_spatial = torch.rot90(vit_spatial, k=3, dims=[2, 3])
                        elif k == 2:  # Was 180°, rotate 180° back
                            vit_spatial = torch.rot90(vit_spatial, k=2, dims=[2, 3])
                        elif k == 3:  # Was 270° CCW, rotate 90° CCW
                            vit_spatial = torch.rot90(vit_spatial, k=1, dims=[2, 3])

                        # Convert back to [B, h, w, C]
                        vit_spatial = vit_spatial.permute(0, 2, 3, 1)
                        rotated_back_list.append(vit_spatial)

                # Stack and average rotated features [B, h, w, C]
                vit_embeds = torch.stack(rotated_back_list, dim=0).mean(dim=0)

                # Apply pixel shuffle (batched operation)
                vit_embeds = self.pixel_shuffle(
                    vit_embeds, scale_factor=self.downsample_ratio
                )

                # Reshape to sequence format [B, N', C]
                vit_embeds = vit_embeds.reshape(
                    vit_embeds.shape[0], -1, vit_embeds.shape[-1]
                )

                # Apply MLP (batched operation)
                vit_embeds = self.teacher_mlp1(vit_embeds)

            else:
                # Original implementation without rotation augmentation
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
        # IMPORTANT: Pass inputs in [-1, 1] range to match student's input
        teacher_features = None
        if self.distillation_weight > 0.0 and self.teacher_vision_model is not None:
            teacher_features = self.extract_teacher_features(
                inputs, use_rotation_aug=self.use_rotation_aug
            )

        # Convert to [0, 1] range for reconstruction and perceptual loss
        inputs = (inputs * 0.5) + 0.5
        reconstructions = (reconstructions * 0.5) + 0.5

        # Convert inputs from [-1, 1] to [0, 1] for loss computation
        # from src.utils.image_utils import normalize_from_neg1_to_1, denormalize_imagenet

        # inputs = normalize_from_neg1_to_1(inputs)

        # # Convert reconstructions from ImageNet normalization to [0, 1]
        # reconstructions = denormalize_imagenet(reconstructions, clamp=True)

        # Compute reconstruction loss (MSE or L1)
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(
                f"Unsupported reconstruction_loss {self.reconstruction_loss}"
            )

        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss (LPIPS)
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute GAN loss
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = (
            self.discriminator_factor
            if self.should_discriminator_be_trained(global_step)
            else 0
        )
        d_weight = 1.0

        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Use discriminator without updating its parameters
            # No need to set requires_grad=False, just don't backward through it
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # Compute distillation loss
        distillation_loss = torch.zeros((), device=inputs.device)
        cosine_loss = torch.zeros((), device=inputs.device)
        mse_loss = torch.zeros((), device=inputs.device)

        if (
            self.distillation_weight > 0.0
            and student_features is not None
            and teacher_features is not None
        ):
            if self.distillation_loss_type == "mse":
                mse_loss = F.mse_loss(
                    student_features, teacher_features, reduction="mean"
                )
                distillation_loss = mse_loss
            elif self.distillation_loss_type == "cosine":
                # Cosine similarity loss (1 - cosine_similarity)
                student_norm = F.normalize(student_features, p=2, dim=-1)
                teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
                cosine_sim = (student_norm * teacher_norm).sum(dim=-1).mean()
                cosine_loss = 1.0 - cosine_sim

                # Also compute MSE loss
                mse_loss = F.mse_loss(
                    student_features, teacher_features, reduction="mean"
                )

                # Combine both losses
                distillation_loss = cosine_loss + mse_loss
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
            distillation_cosine_loss=cosine_loss.detach(),
            mse_loss=mse_loss.detach(),
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
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()

        inputs = (inputs * 0.5) + 0.5
        reconstructions = (reconstructions * 0.5) + 0.5
        # Convert from [-1, 1] to [0, 1]
        # from src.utils.image_utils import normalize_from_neg1_to_1, denormalize_imagenet

        # inputs_01 = normalize_from_neg1_to_1(inputs)

        # # Convert reconstructions from ImageNet normalization to [0, 1]
        # reconstructions_01 = denormalize_imagenet(reconstructions, clamp=True)

        # Discriminator parameters are already trainable by default
        # No need to manually set requires_grad=True
        # real_images = inputs_01  # .detach().requires_grad_(True)
        # logits_real = self.discriminator(real_images)
        # logits_fake = self.discriminator(reconstructions_01.detach())
        logits_real = self.discriminator(inputs)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(
            logits_real=logits_real, logits_fake=logits_fake
        )

        # Optional LeCam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            # 计算 mean
            curr_real_mean = torch.mean(logits_real)
            curr_fake_mean = torch.mean(logits_fake)

            lecam_loss = (
                compute_lecam_loss(
                    curr_real_mean,
                    curr_fake_mean,
                    self.ema_real_logits_mean,
                    self.ema_fake_logits_mean,
                )
                * self.lecam_regularization_weight
            )

            # 关键修改：原地更新 Buffer
            if self.training:  # 只在训练时更新 EMA
                self.ema_real_logits_mean.mul_(self.lecam_ema_decay).add_(
                    curr_real_mean.detach(), alpha=(1 - self.lecam_ema_decay)
                )
                self.ema_fake_logits_mean.mul_(self.lecam_ema_decay).add_(
                    curr_fake_mean.detach(), alpha=(1 - self.lecam_ema_decay)
                )

        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )

        return discriminator_loss, loss_dict
