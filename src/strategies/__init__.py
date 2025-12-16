"""Custom training strategies for PyTorch Lightning."""

from .multi_model_ddp import MultiModelDDPStrategy

__all__ = ["MultiModelDDPStrategy"]
