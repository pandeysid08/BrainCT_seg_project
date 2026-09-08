"""
Metrics used to compare models.

Why each one is here, and what it catches that the others miss:

- Dice coefficient (F1 over pixels): the standard medical-segmentation
  metric. 2*TP / (2*TP + FP + FN). Symmetric, sensitive to small structures.
- IoU / Jaccard: TP / (TP+FP+FN). Monotonic with Dice but harsher (Dice is
  always >= IoU); reported because some papers use one, some the other --
  an interviewer may ask you to convert between them (IoU = Dice/(2-Dice)).
- Precision: of predicted-positive pixels, how many are correct. Low
  precision = model over-segments (false alarms).
- Recall / Sensitivity: of actual-positive pixels, how many were found.
  Low recall = model misses parts of the lesion -- usually the worse
  failure mode clinically.
- Specificity: of actual-negative pixels, how many correctly predicted
  negative. Included because it's cheap and near-1.0 given the huge
  background class, i.e. it demonstrates *why* specificity is a bad
  headline metric here (it barely moves between a good and a broken model).
- 95th-percentile Hausdorff Distance (HD95): boundary-based metric,
  measures worst-case surface distance between predicted and ground-truth
  boundary, robust to the single worst outlier point (unlike max Hausdorff).
  Two masks can have identical Dice but very different HD95 if one has a
  smooth boundary and the other has a stray blob far from the lesion --
  this is the metric that would catch that. Computed only when both masks
  have foreground pixels (undefined for empty masks).
- Per-image inference time and parameter count: reported because "why did
  we pick this model" in a real project is never *purely* about accuracy --
  a 1% Dice gain that costs 5x the inference latency is a real trade-off to
  be able to discuss.
"""
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def _binarize(x, threshold=0.5):
    return (x > threshold).astype(np.uint8)


def dice_coefficient(pred, target, smooth=1e-6):
    pred, target = pred.flatten(), target.flatten()
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-6):
    pred, target = pred.flatten(), target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision_recall_specificity(pred, target, smooth=1e-6):
    pred, target = pred.flatten(), target.flatten()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    return precision, recall, specificity


def hausdorff_distance_95(pred, target):
    """95th percentile symmetric Hausdorff distance in pixels.
    Returns np.nan if either mask is empty (metric undefined)."""
    if pred.sum() == 0 or target.sum() == 0:
        return np.nan

    def surface_points_distance(a, b):
        # distance from every point of `a`'s boundary to nearest point of `b`
        dt = distance_transform_edt(1 - b)
        boundary_a = a.astype(bool)
        return dt[boundary_a]

    d_pred_to_target = surface_points_distance(pred, target)
    d_target_to_pred = surface_points_distance(target, pred)
    all_d = np.concatenate([d_pred_to_target, d_target_to_pred])
    return float(np.percentile(all_d, 95))


def compute_all_metrics(pred_logits: torch.Tensor, target: torch.Tensor, threshold=0.5) -> dict:
    """pred_logits, target: (B,1,H,W) tensors. Returns per-batch-mean metrics dict."""
    probs = torch.sigmoid(pred_logits).detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    preds = _binarize(probs, threshold)

    dices, ious, precs, recs, specs, hd95s = [], [], [], [], [], []
    for i in range(preds.shape[0]):
        p, t = preds[i, 0], target_np[i, 0]
        dices.append(dice_coefficient(p, t))
        ious.append(iou_score(p, t))
        pr, rc, sp = precision_recall_specificity(p, t)
        precs.append(pr); recs.append(rc); specs.append(sp)
        hd95s.append(hausdorff_distance_95(p, t))

    return {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "specificity": float(np.mean(specs)),
        "hd95": float(np.nanmean(hd95s)) if not np.all(np.isnan(hd95s)) else np.nan,
    }
