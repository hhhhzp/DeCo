import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_info
from src.models.autoencoder.base import fp2uint8
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
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
        # 初始化指标计算器，dist_sync_on_step=False 表示我们只在 epoch 结束时同步
        # data_range=1.0 对应 [0, 1] 的数据范围
        self.psnr = PeakSignalNoiseRatio(data_range=(0, 255.0))
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(0, 255.0))

        # FID 相关
        self.compute_fid = compute_fid
        self.fid_feature_dim = fid_feature_dim
        if compute_fid:
            # 使用 TorchMetrics 的 FID，它内部已经处理了 DDP 同步
            self.fid = FrechetInceptionDistance(
                feature=fid_feature_dim, antialias=False
            )

        # 用于增量统计的变量
        self.reset_fid_stats()

    def reset_fid_stats(self):
        """Reset FID statistics for a new epoch"""
        self.fid_enabled = False

    def setup(self, trainer, pl_module, stage=None):
        # 确保指标模块被移动到正确的设备上
        if hasattr(pl_module, "device"):
            self.psnr = self.psnr.to(pl_module.device)
            self.ssim = self.ssim.to(pl_module.device)
            if self.compute_fid:
                self.fid = self.fid.to(pl_module.device)

    def _update_metrics(self, pl_module, outputs, batch):
        # 提取数据 (假设 batch 格式为 [img, label, metadata])
        original_img, _, _ = batch

        # 归一化处理
        reconstructed = fp2uint8(outputs)
        original_img = fp2uint8(original_img)

        # 更新 PSNR 和 SSIM (此时不计算最终值，只累积统计量)
        self.psnr.update(reconstructed, original_img)
        self.ssim.update(reconstructed, original_img)

        # 更新 FID (如果启用)
        if self.compute_fid and self.fid_enabled:
            # 将图像转换为 uint8 [0, 255] 格式，这是 FID 期望的输入
            # reconstructed_uint8 = fp2uint8(reconstructed)
            # original_uint8 = fp2uint8(original_img)

            # TorchMetrics FID 会自动提取 Inception 特征并累积统计量
            # 这里只在每个 rank 上处理自己的 batch，不会 OOM
            self.fid.update(original_img, real=True)
            self.fid.update(reconstructed, real=False)

    def _log_and_reset(self, pl_module, prefix="val"):
        # 计算最终指标 (自动处理 DDP 同步)
        final_psnr = self.psnr.compute()
        final_ssim = self.ssim.compute()

        log_msg = f"[{prefix}] PSNR: {final_psnr:.4f} | SSIM: {final_ssim:.4f}"

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
        self.psnr.reset()
        self.ssim.reset()
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
