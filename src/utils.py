"""
Small, boring, but important utilities.
Interview point: reproducibility (seeding every RNG source) is something
interviewers specifically probe for, because it's the #1 thing people forget.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Seed python, numpy and torch (CPU + CUDA) RNGs, and force
    deterministic cuDNN kernels. Deterministic cuDNN is slightly slower
    but makes results reproducible run-to-run -- important when you are
    comparing 5 architectures and need the differences to be due to the
    architecture, not to random seed noise.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AverageMeter:
    """Tracks a running mean. Used for loss / metric logging per epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


class EarlyStopping:
    """Stops training when the monitored metric stops improving.

    Why this matters for THIS project specifically: with only ~100 images,
    a model can memorize the training set within a handful of epochs.
    Early stopping on a validation Dice score is the main defence against
    overfitting, alongside augmentation and weight decay.
    """

    def __init__(self, patience: int = 15, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Returns True if `value` is the new best score."""
        if self.best is None:
            self.best = value
            return True
        improved = (value > self.best + self.min_delta) if self.mode == "max" \
            else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
