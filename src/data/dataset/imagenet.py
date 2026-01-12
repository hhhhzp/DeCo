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
        root,
        split='train',
        resolution=256,
        random_crop=False,
        random_flip=False,
        max_num_samples=None,
    ):
        """
        HuggingFace Dataset wrapper with flexible train/validation split support.

        Args:
            root: HuggingFace dataset name or path
            split: Dataset split ('train' or 'validation')
            resolution: Target image resolution
            random_crop: Whether to use random crop (for training)
            random_flip: Whether to use random horizontal flip (for training)
            max_num_samples: Maximum number of samples to use
        """
        super().__init__()

        self.split = split
        self.dataset = load_dataset(root, split=split, trust_remote_code=True)

        # Handle max_num_samples
        if max_num_samples is not None and max_num_samples < len(self.dataset):
            total_samples = len(self.dataset)
            np.random.seed(42)
            selected_indices = np.random.choice(
                total_samples,
                size=min(max_num_samples, total_samples),
                replace=False,
            ).tolist()
            self.dataset = self.dataset.select(selected_indices)
            print(f"Subsampled {split} split to {len(self.dataset)} samples")

        # Setup transforms using create_image_transform function
        # For validation split, force random_crop=False and random_flip=False
        is_train = split == 'train'
        self.transform = create_image_transform(
            resolution=resolution,
            random_crop=random_crop if is_train else False,
            random_flip=random_flip if is_train else False,
            normalize_mean=[0.5, 0.5, 0.5],
            normalize_std=[0.5, 0.5, 0.5],
        )
        print(f"PixHFDataset ({split}) transform: {self.transform}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # 1. Get item from HF Dataset
        item = self.dataset[idx]

        # 2. Process image (ensure PIL RGB)
        raw_image = item['image']
        if raw_image.mode != 'RGB':
            raw_image = raw_image.convert('RGB')

        target = item['label']

        # 3. Apply transform (includes Resize -> Crop -> Flip -> ToTensor -> Normalize)
        normalized_image = self.transform(raw_image)

        # 4. Construct metadata
        metadata = {
            "raw_image": normalized_image,  # Normalized tensor [-1, 1]
            "class": target,
        }
        return normalized_image, target, metadata


import orjson
import os


class PixJSONLDataset(Dataset):
    def __init__(
        self,
        root,
        annotation,
        resolution=256,
        random_crop=False,
        random_flip=False,
        max_num_samples=None,
    ):
        """
        Dataset for JSONL format data.

        Args:
            root: Root directory containing the images
            annotation: Path to the JSONL annotation file
            resolution: Image resolution
            random_crop: Whether to use random crop
            random_flip: Whether to use random horizontal flip
            max_num_samples: Maximum number of samples to use (for subset evaluation)
        """
        super().__init__()

        self.root = root
        self.resolution = resolution

        # Load JSONL annotations with orjson for faster parsing
        self.samples = []
        with open(annotation, 'rb') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = orjson.loads(line)
                    if 'target_image' in item:
                        self.samples.append(item)

        # Handle max_num_samples
        if max_num_samples is not None and max_num_samples < len(self.samples):
            np.random.seed(42)
            selected_indices = np.random.choice(
                len(self.samples),
                size=min(max_num_samples, len(self.samples)),
                replace=False,
            ).tolist()
            self.samples = [self.samples[i] for i in selected_indices]

        # Setup transforms
        if random_crop:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(resolution),
                    transforms.RandomCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
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
        return len(self.samples)

    def __getitem__(self, idx: int):
        max_retries = 5  # 最大重试次数

        for attempt in range(max_retries):
            try:
                # 如果是重试，随机选择另一个索引
                if attempt > 0:
                    idx = np.random.randint(0, len(self.samples))

                # 1. Get sample metadata
                item = self.samples[idx]

                # 2. Construct image path
                image_rel_path = item['target_image']
                image_path = os.path.join(self.root, image_rel_path)

                # 3. Load and process image
                raw_image = Image.open(image_path).convert('RGB')

                # 4. Apply transforms
                raw_image = self.transform(raw_image)

                # 5. Convert to tensor
                raw_image = to_tensor(raw_image)

                # 6. Normalize
                normalized_image = self.normalize(raw_image)

                # 7. Set target to 0 (as requested)
                target = 0

                # 8. Construct metadata
                metadata = {
                    "raw_image": raw_image,  # Unnormalized tensor (0-1)
                    "class": target,
                }

                return normalized_image, target, metadata

            except Exception as e:
                # 如果是最后一次尝试，抛出异常
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to load image after {max_retries} attempts. Last error: {e}"
                    )
                # 否则继续重试
                continue


def create_image_transform(
    resolution=256,
    random_crop=False,
    random_flip=False,
    normalize_mean=[0.5, 0.5, 0.5],
    normalize_std=[0.5, 0.5, 0.5],
):
    """
    Create image transformation pipeline.

    Args:
        resolution: Target image resolution
        random_crop: Whether to use random crop (otherwise center crop)
        random_flip: Whether to use random horizontal flip
        normalize_mean: Mean values for normalization
        normalize_std: Std values for normalization

    Returns:
        A composed transform that includes Resize -> Crop -> Flip -> ToTensor -> Normalize
    """
    train_transform = []
    interpolation = transforms.InterpolationMode.BICUBIC

    # Resize shorter edge
    train_transform.append(
        transforms.Resize(resolution, interpolation=interpolation, antialias=True)
    )

    # Crop
    if random_crop:
        train_transform.append(transforms.RandomCrop(resolution))
    else:
        train_transform.append(transforms.CenterCrop(resolution))

    # Random flip
    if random_flip:
        train_transform.append(transforms.RandomHorizontalFlip())

    # ToTensor
    train_transform.append(transforms.ToTensor())

    # Normalize
    train_transform.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))

    return transforms.Compose(train_transform)


class PixMultiJSONLDataset(Dataset):
    """
    Dataset that loads multiple JSONL datasets from a JSON config file.
    Supports merging multiple datasets with repeat_time parameter.
    """

    def __init__(
        self,
        config_path,
        resolution=256,
        random_crop=False,
        random_flip=False,
        max_num_samples=None,
    ):
        """
        Args:
            config_path: Path to JSON config file (e.g., scripts/total_images.json)
            resolution: Image resolution
            random_crop: Whether to use random crop
            random_flip: Whether to use random horizontal flip
            max_num_samples: Maximum number of samples to use (applied after merging)
        """
        super().__init__()

        self.resolution = resolution
        self.config_path = config_path

        # Load config file with orjson for faster parsing
        with open(config_path, 'rb') as f:
            config = orjson.loads(f.read())

        # Collect all samples from all datasets
        self.samples = []
        self.dataset_names = []

        for dataset_name, dataset_config in config.items():
            root = dataset_config['root']
            annotation = dataset_config['annotation']
            repeat_time = dataset_config.get('repeat_time', 1)

            # Load samples from this dataset with orjson for faster parsing
            dataset_samples = []
            with open(annotation, 'rb') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = orjson.loads(line)
                        if 'target_image' in item:
                            # Add root path to each sample
                            sample = {
                                'root': root,
                                'target_image': item['target_image'],
                                'dataset_name': dataset_name,
                            }
                            dataset_samples.append(sample)

            # Repeat samples according to repeat_time
            for _ in range(repeat_time):
                self.samples.extend(dataset_samples)

            print(
                f"Loaded {len(dataset_samples)} samples from {dataset_name} "
                f"(repeated {repeat_time} times, total: {len(dataset_samples) * repeat_time})"
            )

        print(f"Total samples after merging: {len(self.samples)}")

        # Handle max_num_samples
        if max_num_samples is not None and max_num_samples < len(self.samples):
            np.random.seed(42)
            selected_indices = np.random.choice(
                len(self.samples),
                size=min(max_num_samples, len(self.samples)),
                replace=False,
            ).tolist()
            self.samples = [self.samples[i] for i in selected_indices]
            print(f"Subsampled to {len(self.samples)} samples")

        # Setup transforms using the create_image_transform function
        self.transform = create_image_transform(
            resolution=resolution,
            random_crop=random_crop,
            random_flip=random_flip,
            normalize_mean=[0.5, 0.5, 0.5],
            normalize_std=[0.5, 0.5, 0.5],
        )
        print(f"PixMultiJSONLDataset transform: {self.transform}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        max_retries = 5  # 最大重试次数

        for attempt in range(max_retries):
            try:
                # 如果是重试，随机选择另一个索引
                if attempt > 0:
                    idx = np.random.randint(0, len(self.samples))

                # 1. Get sample metadata
                item = self.samples[idx]
                root = item['root']
                image_rel_path = item['target_image']

                # 2. Construct image path
                image_path = os.path.join(root, image_rel_path)

                # 3. Load and process image
                raw_image = Image.open(image_path).convert('RGB')

                # 4. Apply transforms (includes ToTensor and Normalize)
                normalized_image = self.transform(raw_image)

                # 5. Set target to 0 (as requested)
                target = 0

                # 6. Construct metadata
                # Note: raw_image is now the normalized tensor, so we store it as is
                metadata = {
                    "raw_image": normalized_image,  # Normalized tensor [-1, 1]
                    "class": target,
                    "dataset_name": item['dataset_name'],
                }

                return normalized_image, target, metadata

            except Exception as e:
                # 如果是最后一次尝试，抛出异常
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to load image after {max_retries} attempts. "
                        f"Last error: {e}, Image path: {image_path if 'image_path' in locals() else 'unknown'}"
                    )
                # 否则继续重试
                continue


import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset


class PixWebDataset(IterableDataset):
    """
    WebDataset wrapper with streaming support and distributed training.
    Supports automatic sharding for multi-GPU and multi-worker scenarios.
    """

    def __init__(
        self,
        data_files,
        cache_dir=None,
        resolution=256,
        random_crop=False,
        random_flip=False,
        is_train=True,
        random_seed=42,
    ):
        """
        Args:
            data_files: List of WebDataset tar files or pattern
            cache_dir: Cache directory for HuggingFace datasets
            resolution: Image resolution
            random_crop: Whether to use random crop
            random_flip: Whether to use random horizontal flip
            is_train: Whether this is training dataset (affects shuffling)
            data_rank: Current process rank in distributed training
            data_world_size: Total number of processes in distributed training
            random_seed: Random seed for shuffling
        """
        self.data_files = data_files
        self.cache_dir = cache_dir
        self.resolution = resolution
        self.is_train = is_train
        self.random_seed = random_seed

        # Setup transforms (similar to PixImageNet)
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

        # Load HuggingFace WebDataset
        meta = {"cache_dir": cache_dir} if cache_dir else {}
        self.hf_dataset = load_dataset(
            "webdataset",
            data_files=data_files,
            cache_dir=meta.get("cache_dir"),
            split="train",
            streaming=True,
        )

        # Setup sharding for distributed training
        self.hf_dataset = self._setup_sharded_dataset(self.hf_dataset)

        print(f"PixWebDataset initialized with transform: {self.transform}")

    def _process_sample(self, sample):
        """Process a single sample from WebDataset"""
        # Extract image from sample
        # WebDataset format typically has keys like 'jpg', 'png', 'image', etc.
        image = sample['jpg']
        image = image.convert('RGB')

        # Apply transforms
        raw_image = self.transform(image)
        raw_image = to_tensor(raw_image)

        # Normalize
        normalized_image = self.normalize(raw_image)

        # Extract label if available
        target = 0

        # Construct metadata
        metadata = {
            "raw_image": raw_image,  # Unnormalized tensor (0-1)
            "class": target,
        }

        return normalized_image, target, metadata

    def _setup_sharded_dataset(self, dataset):
        """
        处理两级分片逻辑：
        1. Global Level: 根据 GPU rank 切分
        2. Local Level:  根据 DataLoader worker 切分
        """

        # --- 1. Global Sharding (GPU/Node 级别) ---
        # 自动检测是否处于分布式环境
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        # 如果是多卡训练，先进行第一层切分
        if world_size > 1:
            dataset = dataset.shard(num_shards=world_size, index=rank)

        # [可选] 这里是插入 Shuffle 的最佳时机
        if self.is_train:
            dataset = dataset.shuffle(buffer_size=1000, seed=self.seed)

        # --- 2. Local Sharding (Worker 级别) ---
        # 获取当前进程的 worker 信息
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            # 在当前 GPU 分到的数据基础上，再进行第二层切分
            dataset = dataset.shard(
                num_shards=worker_info.num_workers, index=worker_info.id
            )

        return dataset

    def __iter__(self):
        """Iterate over the dataset"""
        dataset = self._setup_sharded_dataset(self.hf_dataset)
        for sample in dataset:
            try:
                yield self._process_sample(sample)
            except Exception as e:
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(f"[Rank {rank}] Warning: Sample failed: {e}")
                continue
