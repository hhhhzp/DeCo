"""Image normalization and denormalization utilities."""

import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def denormalize_imagenet(
    images: torch.Tensor,
    mean: tuple = IMAGENET_DEFAULT_MEAN,
    std: tuple = IMAGENET_DEFAULT_STD,
    clamp: bool = True,
) -> torch.Tensor:
    """
    Denormalize images from ImageNet normalization to [0, 1] range.

    Args:
        images: Input tensor with ImageNet normalization, shape [B, C, H, W]
        mean: Mean values used for normalization (default: ImageNet mean)
        std: Std values used for normalization (default: ImageNet std)
        clamp: Whether to clamp output to [0, 1] range

    Returns:
        Denormalized images in [0, 1] range
    """
    # Convert mean and std to tensors with correct shape [1, C, 1, 1]
    mean_tensor = torch.tensor(mean, device=images.device, dtype=images.dtype).view(
        1, -1, 1, 1
    )
    std_tensor = torch.tensor(std, device=images.device, dtype=images.dtype).view(
        1, -1, 1, 1
    )

    # Denormalize: x_original = x_normalized * std + mean
    denormalized = images * std_tensor + mean_tensor

    if clamp:
        denormalized = torch.clamp(denormalized, 0.0, 1.0)

    return denormalized


def normalize_from_neg1_to_1(images: torch.Tensor) -> torch.Tensor:
    """
    Convert images from [-1, 1] range to [0, 1] range.

    Args:
        images: Input tensor in [-1, 1] range, shape [B, C, H, W]

    Returns:
        Images in [0, 1] range
    """
    return (images + 1.0) / 2.0


def normalize_to_neg1_to_1(images: torch.Tensor) -> torch.Tensor:
    """
    Convert images from [0, 1] range to [-1, 1] range.

    Args:
        images: Input tensor in [0, 1] range, shape [B, C, H, W]

    Returns:
        Images in [-1, 1] range
    """
    return images * 2.0 - 1.0


def denormalize_to_uint8(
    images: torch.Tensor,
    source_range: str = "imagenet",
    mean: tuple = IMAGENET_DEFAULT_MEAN,
    std: tuple = IMAGENET_DEFAULT_STD,
) -> torch.Tensor:
    """
    Denormalize images and convert to uint8 [0, 255] range.

    Args:
        images: Input tensor, shape [B, C, H, W]
        source_range: Source normalization type, one of:
            - "imagenet": ImageNet normalization (mean/std)
            - "neg1_to_1": Range [-1, 1]
            - "0_to_1": Range [0, 1]
        mean: Mean values for ImageNet denormalization
        std: Std values for ImageNet denormalization

    Returns:
        Images in uint8 [0, 255] range
    """
    if source_range == "imagenet":
        # Denormalize from ImageNet to [0, 1]
        images_01 = denormalize_imagenet(images, mean, std, clamp=True)
    elif source_range == "neg1_to_1":
        # Convert from [-1, 1] to [0, 1]
        images_01 = normalize_from_neg1_to_1(images)
        images_01 = torch.clamp(images_01, 0.0, 1.0)
    elif source_range == "0_to_1":
        # Already in [0, 1], just clamp
        images_01 = torch.clamp(images, 0.0, 1.0)
    else:
        raise ValueError(
            f"Unknown source_range: {source_range}. "
            f"Must be one of: 'imagenet', 'neg1_to_1', '0_to_1'"
        )

    # Convert to [0, 255] uint8
    images_uint8 = (images_01 * 255.0).to(torch.uint8)

    return images_uint8
