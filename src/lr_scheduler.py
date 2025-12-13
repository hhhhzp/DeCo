import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """Custom lr_scheduler that combines warm up with cosine annealing.

    Args:
        optimizer: The optimizer to schedule
        T_max: Maximum number of iterations for cosine annealing
        eta_min: Minimum learning rate
        warmup_steps: Number of warm up steps
        warmup_start_lr: Starting learning rate for warm up (default: 0.01 * base_lr)
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=0,
        warmup_steps=1000,
        warmup_start_lr=None,
        last_epoch=-1,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr

        # Initialize parent class first to set base_lrs
        super().__init__(optimizer, last_epoch)

        # Set warmup start lr if not provided
        if self.warmup_start_lr is None:
            self.warmup_start_lr = [base_lr * 0.01 for base_lr in self.base_lrs]

        # Initialize cosine scheduler for after warmup
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=T_max - warmup_steps, eta_min=eta_min
        )

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warm up phase: linear increase from warmup_start_lr to base_lr
            progress = self.last_epoch / self.warmup_steps
            return [
                start_lr + progress * (base_lr - start_lr)
                for start_lr, base_lr in zip(self.warmup_start_lr, self.base_lrs)
            ]
        else:
            # Cosine annealing phase
            # Adjust epoch for cosine scheduler
            cosine_epoch = self.last_epoch - self.warmup_steps
            self.cosine_scheduler.last_epoch = cosine_epoch
            return self.cosine_scheduler.get_lr()
