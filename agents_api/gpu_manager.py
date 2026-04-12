"""GPU resource manager for ML training.

Thread-safe context manager that controls GPU allocation for training.
Configurable via SVEND_GPU_TRAINING_PERCENT env var (default: 50%).
Falls back to CPU if GPU is busy or unavailable.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

_gpu_lock = threading.Lock()
_GPU_PERCENT = int(os.environ.get("SVEND_GPU_TRAINING_PERCENT", "50"))


class GPUTrainingContext:
    """Context manager for GPU-accelerated training.

    Usage:
        with GPUTrainingContext() as gpu:
            model = xgb.XGBRegressor(**gpu.xgb_params())
            model.fit(X, y)
    """

    def __init__(self):
        self.available = False
        self._acquired = False

    def __enter__(self):
        if _gpu_lock.acquire(blocking=False):
            self._acquired = True
            try:
                import torch

                if torch.cuda.is_available():
                    self.available = True
                    logger.info(f"GPU acquired for training ({_GPU_PERCENT}% allocation)")
                else:
                    logger.info("CUDA not available, falling back to CPU")
            except ImportError:
                logger.info("PyTorch not installed, falling back to CPU")
        else:
            logger.info("GPU busy (another training in progress), falling back to CPU")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._acquired:
            _gpu_lock.release()
            self._acquired = False
            if self.available:
                logger.info("GPU released after training")
        return False

    def xgb_params(self):
        """XGBoost GPU parameters."""
        if self.available:
            return {"tree_method": "hist", "device": "cuda"}
        return {"tree_method": "hist"}

    def lgb_params(self):
        """LightGBM GPU parameters."""
        if self.available:
            return {"device": "gpu"}
        return {}
