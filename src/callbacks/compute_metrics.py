import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_info
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class ComputeMetricsHook(Callback):
    """
    Callback for computing reconstruction quality metrics (PSNR, SSIM) using TorchMetrics.
    Fully GPU-accelerated and DDP-safe.
    """

    def __init__(self):
        super().__init__()
        # 初始化指标计算器，dist_sync_on_step=False 表示我们只在 epoch 结束时同步
        # data_range=1.0 对应 [0, 1] 的数据范围
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def setup(self, trainer, pl_module, stage=None):
        # 确保指标模块被移动到正确的设备上
        if hasattr(pl_module, "device"):
            self.psnr = self.psnr.to(pl_module.device)
            self.ssim = self.ssim.to(pl_module.device)

    def _update_metrics(self, pl_module, outputs, batch):
        # 提取数据 (假设 batch 格式为 [img, label, metadata])
        original_img, _, _ = batch

        # 确保数据在同一设备
        reconstructed = outputs.to(pl_module.device)
        original_img = original_img.to(pl_module.device)

        # 根据数据类型进行归一化处理
        if reconstructed.dtype == torch.uint8:
            # uint8 类型: [0, 255] -> [0, 1]
            reconstructed = reconstructed.float() / 255.0
        else:
            # float 类型: 假设 [-1, 1] -> [0, 1]
            reconstructed = torch.clamp((reconstructed + 1.0) / 2.0, 0.0, 1.0)

        if original_img.dtype == torch.uint8:
            # uint8 类型: [0, 255] -> [0, 1]
            original_img = original_img.float() / 255.0
        else:
            # float 类型: 假设 [-1, 1] -> [0, 1]
            original_img = torch.clamp((original_img + 1.0) / 2.0, 0.0, 1.0)

        # 更新指标状态 (此时不计算最终值，只累积统计量)
        self.psnr.update(reconstructed, original_img)
        self.ssim.update(reconstructed, original_img)

    def _log_and_reset(self, pl_module, prefix="val"):
        # 计算最终指标 (自动处理 DDP 同步)
        final_psnr = self.psnr.compute()
        final_ssim = self.ssim.compute()

        rank_zero_info(f"[{prefix}] PSNR: {final_psnr:.4f} | SSIM: {final_ssim:.4f}")

        # Log 到 TensorBoard/WandB
        pl_module.log(f"{prefix}/psnr", final_psnr, sync_dist=True, rank_zero_only=True)
        pl_module.log(f"{prefix}/ssim", final_ssim, sync_dist=True, rank_zero_only=True)

        # 重置状态以备下一个 epoch
        self.psnr.reset()
        self.ssim.reset()

    # ========== Validation hooks ==========
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._update_metrics(pl_module, outputs, batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_and_reset(pl_module, prefix="metrics")  # 或者 "val"

    # ========== Prediction hooks ==========
    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._update_metrics(pl_module, outputs, batch)

    def on_predict_epoch_end(self, trainer, pl_module):
        self._log_and_reset(pl_module, prefix="predict_metrics")
