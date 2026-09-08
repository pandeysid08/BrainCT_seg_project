"""
Loss functions.

Interview-critical point: pixel-wise class imbalance. A hemorrhage / lesion
region is often <5% of brain pixels, sometimes <1%. Plain BCE is dominated
by the easy background class -- a model predicting all-zero can still get
95%+ pixel accuracy while being clinically useless (Dice = 0). This is why
segmentation benchmarks report Dice/IoU, not accuracy, and why the loss
function itself is usually a Dice-family loss, or a Dice+BCE combo.

We implement:
  - DiceLoss: directly optimizes the overlap metric we will be evaluated on.
  - BCEDiceLoss: BCE gives per-pixel gradient signal even when overlap is
    zero early in training (Dice loss's gradient is unstable/near-zero when
    the model predicts nothing on the foreground yet); Dice supplies the
    overlap-aware, imbalance-robust signal. This combination is the most
    common default in medical segmentation.
  - TverskyLoss / FocalTverskyLoss: generalizes Dice with separate weights
    on false positives vs false negatives (alpha, beta). In a screening
    context (e.g. "did this scan have a bleed") missing a lesion (FN) is
    worse than a spurious extra pixel (FP), so beta > alpha lets us penalize
    FN harder -- something plain Dice/BCE cannot express.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + (1 - self.bce_weight) * self.dice(logits, targets)


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, smooth: float = 1.0):
        # beta > alpha => false negatives penalized more than false positives.
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        tp = (probs * targets).sum(dim=1)
        fp = ((1 - targets) * probs).sum(dim=1)
        fn = (targets * (1 - probs)).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return ((1 - tversky) ** self.gamma).mean()


def get_loss(name: str):
    return {
        "dice": DiceLoss(),
        "bce_dice": BCEDiceLoss(),
        "focal_tversky": FocalTverskyLoss(),
        "bce": nn.BCEWithLogitsLoss(),
    }[name]
