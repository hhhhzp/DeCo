import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_info
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np


class ComputeMetricsHook(Callback):
    """
    Callback for computing reconstruction quality metrics (PSNR, SSIM, FID) using TorchMetrics.
    Fully GPU-accelerated and DDP-safe.

    FID is computed by collaborating with SaveImagesHook to avoid OOM:
    - Each rank processes its own batch and extracts Inception features
    - Features are accumulated incrementally (mean and covariance statistics)
    - Only statistics are synchronized across ranks, not raw features
    """

    def __init__(self, compute_fid=True, fid_feature_dim=2048):
        super().__init__()
        # 使用 scikit-image 计算 PSNR 和 SSIM，需要手动累积结果
        self.psnr_values = []
        self.ssim_values = []

        # FID 相关
        self.compute_fid = compute_fid
        self.fid_feature_dim = fid_feature_dim
        if compute_fid:
            # 使用 TorchMetrics 的 FID，它内部已经处理了 DDP 同步
            self.fid = FrechetInceptionDistance(
                feature=fid_feature_dim, normalize=True, antialias=False
            )

        # 用于增量统计的变量
        self.reset_fid_stats()

    def reset_fid_stats(self):
        """Reset FID statistics for a new epoch"""
        self.fid_enabled = False

    def setup(self, trainer, pl_module, stage=None):
        # 确保指标模块被移动到正确的设备上
        if hasattr(pl_module, "device"):
            if self.compute_fid:
                self.fid = self.fid.to(pl_module.device)

    def _normalize_images(self, images, device):
        """Normalize images to [0, 1] range"""
        from src.utils.image_utils import normalize_from_neg1_to_1

        images = images.to(device)
        if images.dtype == torch.uint8:
            # uint8 类型: [0, 255] -> [0, 1]
            images = images.float() / 255.0
        else:
            # float 类型: 假设 [-1, 1] -> [0, 1]
            images = normalize_from_neg1_to_1(images)
            images = torch.clamp(images, 0.0, 1.0)
        return images

    def _update_metrics(self, pl_module, outputs, batch):
        # 提取数据 (假设 batch 格式为 [img, label, metadata])
        original_img, _, _ = batch

        # 归一化处理
        reconstructed = self._normalize_images(outputs, pl_module.device)
        original_img = self._normalize_images(original_img, pl_module.device)

        # 转换为 numpy 格式 [B, H, W, C]，范围 [0, 1]
        rgb_restored = reconstructed.permute(0, 2, 3, 1).cpu().numpy()
        rgb_gt = original_img.permute(0, 2, 3, 1).cpu().numpy()

        # 逐图像计算 PSNR 和 SSIM
        for rgb_real, rgb_fake in zip(rgb_gt, rgb_restored):
            psnr = psnr_loss(rgb_fake, rgb_real, data_range=1.0)
            ssim = ssim_loss(
                rgb_fake, rgb_real, multichannel=True, data_range=1.0, channel_axis=-1
            )
            self.psnr_values.append(psnr)
            self.ssim_values.append(ssim)

        # 更新 FID (如果启用)
        if self.compute_fid and self.fid_enabled:
            # 将图像转换为 uint8 [0, 255] 格式，这是 FID 期望的输入
            reconstructed_uint8 = (reconstructed * 255).to(torch.uint8)
            original_uint8 = (original_img * 255).to(torch.uint8)

            # TorchMetrics FID 会自动提取 Inception 特征并累积统计量
            # 这里只在每个 rank 上处理自己的 batch，不会 OOM
            self.fid.update(original_uint8, real=True)
            self.fid.update(reconstructed_uint8, real=False)

    def _log_and_reset(self, pl_module, prefix="val"):
        # 计算本地平均值
        local_psnr = np.mean(self.psnr_values) if self.psnr_values else 0.0
        local_ssim = np.mean(self.ssim_values) if self.ssim_values else 0.0
        local_count = len(self.psnr_values)

        # 转换为 tensor 以便跨进程同步
        psnr_tensor = torch.tensor(local_psnr, device=pl_module.device)
        ssim_tensor = torch.tensor(local_ssim, device=pl_module.device)
        count_tensor = torch.tensor(local_count, device=pl_module.device)

        # 在多卡环境下同步所有进程的指标
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # 收集所有进程的值
            torch.distributed.all_reduce(psnr_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(ssim_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                count_tensor, op=torch.distributed.ReduceOp.SUM
            )

            # 计算全局平均值
            world_size = torch.distributed.get_world_size()
            final_psnr = (psnr_tensor / world_size).item()
            final_ssim = (ssim_tensor / world_size).item()
            total_count = count_tensor.item()
        else:
            # 单卡情况
            final_psnr = local_psnr
            final_ssim = local_ssim
            total_count = local_count

        log_msg = f"[{prefix}] PSNR: {final_psnr:.4f} | SSIM: {final_ssim:.4f} | Samples: {total_count}"

        # 计算 FID (如果启用)
        if self.compute_fid and self.fid_enabled:
            try:
                # FID.compute() 会自动在所有 rank 间同步统计量并计算最终 FID
                final_fid = self.fid.compute()
                log_msg += f" | FID: {final_fid:.4f}"
                pl_module.log(
                    f"{prefix}/fid", final_fid, sync_dist=True, rank_zero_only=True
                )
            except Exception as e:
                rank_zero_info(f"FID computation failed: {e}")

        rank_zero_info(log_msg)

        # Log 到 TensorBoard/WandB
        pl_module.log(f"{prefix}/psnr", final_psnr, sync_dist=True, rank_zero_only=True)
        pl_module.log(f"{prefix}/ssim", final_ssim, sync_dist=True, rank_zero_only=True)

        # 重置状态以备下一个 epoch
        self.psnr_values = []
        self.ssim_values = []
        if self.compute_fid:
            self.fid.reset()
        self.reset_fid_stats()

    # ========== Validation hooks ==========
    def on_validation_epoch_start(self, trainer, pl_module):
        # 启用 FID 计算
        self.fid_enabled = self.compute_fid

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._update_metrics(pl_module, outputs, batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_and_reset(pl_module, prefix="metrics")  # 或者 "val"

    # ========== Prediction hooks ==========
    def on_predict_epoch_start(self, trainer, pl_module):
        # 启用 FID 计算
        self.fid_enabled = self.compute_fid

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._update_metrics(pl_module, outputs, batch)

    def on_predict_epoch_end(self, trainer, pl_module):
        self._log_and_reset(pl_module, prefix="predict_metrics")
