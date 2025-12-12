import torch
import torchvision.transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Normalize
from functools import partial

import numpy as np


def center_crop_fn(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


class LocalCachedDataset(ImageFolder):
    def __init__(self, root, resolution=256, cache_root=None):
        super().__init__(root)
        self.cache_root = cache_root
        self.transform = partial(center_crop_fn, image_size=resolution)

    def load_latent(self, latent_path):
        pk_data = torch.load(latent_path)
        mean = pk_data['mean'].to(torch.float32)
        logvar = pk_data['logvar'].to(torch.float32)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        latent = mean + torch.randn_like(mean) * std
        return latent

    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]
        latent_path = image_path.replace(self.root, self.cache_root) + ".pt"

        raw_image = Image.open(image_path).convert('RGB')
        raw_image = self.transform(raw_image)
        raw_image = to_tensor(raw_image)
        if self.cache_root is not None:
            latent = self.load_latent(latent_path)
        else:
            latent = raw_image

        metadata = {
            "raw_image": raw_image,
            "class": target,
        }
        return latent, target, metadata


class PixImageNet(ImageFolder):
    def __init__(self, root, resolution=256, random_crop=False, random_flip=False):
        super().__init__(root)
        if random_crop:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(resolution),
                    torchvision.transforms.RandomCrop(resolution),
                    torchvision.transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            if random_flip is False:
                self.transform = partial(center_crop_fn, image_size=resolution)
            else:
                self.transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Lambda(
                            partial(center_crop_fn, image_size=resolution)
                        ),
                        torchvision.transforms.RandomHorizontalFlip(),
                    ]
                )

        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]
        raw_image = Image.open(image_path).convert('RGB')
        raw_image = self.transform(raw_image)
        raw_image = to_tensor(raw_image)

        normalized_image = self.normalize(raw_image)

        metadata = {
            "raw_image": raw_image,
            "class": target,
        }
        return normalized_image, target, metadata


import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
from datasets import load_dataset


class PixHFDataset(Dataset):
    def __init__(
        self,
        data_path,
        split='train',
        resolution=256,
        random_crop=False,
        random_flip=False,
    ):
        super().__init__()

        # 1. 在类内部完成 load_dataset
        print(f"Loading HF dataset: {data_path} ({split})...")
        self.dataset = load_dataset(data_path, split=split, trust_remote_code=True)

        # 2. 恢复原本的 Transform 逻辑
        if random_crop:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(resolution),
                    transforms.RandomCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            # 完全复用你原来的逻辑
            if random_flip is False:
                self.transform = partial(center_crop_fn, image_size=resolution)
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.Lambda(
                            partial(center_crop_fn, image_size=resolution)
                        ),
                        transforms.RandomHorizontalFlip(),
                    ]
                )

        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # 1. 从 HF Dataset 获取 item
        item = self.dataset[idx]

        # 2. 处理图片 (确保是 PIL RGB)
        raw_image = item['image']
        if raw_image.mode != 'RGB':
            raw_image = raw_image.convert('RGB')

        target = item['label']

        # 3. 执行 Transform (PIL -> PIL)
        # 注意：这里的 self.transform 返回的是 PIL，保持与你原代码一致
        raw_image = self.transform(raw_image)

        # 4. 转 Tensor (PIL -> Tensor)
        raw_image = TF.to_tensor(raw_image)

        # 5. 归一化
        normalized_image = self.normalize(raw_image)

        # 6. 构造 Metadata
        metadata = {
            "raw_image": raw_image,  # 这里是未归一化的 Tensor (0-1)
            "class": target,
        }
        return normalized_image, target, metadata
