"""
Evaluate every trained checkpoint (plus the classical Otsu baseline) on the
SAME frozen test split, and produce a single comparison table + bar chart.

This script is the deliverable an interviewer actually wants to see: a
reproducible, apples-to-apples comparison across architectures with a
justified "we picked X because Y" conclusion at the end.

Usage:
    python src/evaluate.py --results_dir ../results --images_dir ../data/images --masks_dir ../data/masks
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import build_file_pairs, BrainCTSegDataset
from augmentations import get_val_transforms
from metrics import compute_all_metrics, dice_coefficient, iou_score, precision_recall_specificity, hausdorff_distance_95
from models import build_model, count_parameters, otsu_baseline_predict
from utils import get_device


def evaluate_trained_model(model_name, ckpt_path, test_pairs, image_size, device):
    imagenet_norm = model_name != "unet_scratch"
    test_ds = BrainCTSegDataset(test_pairs, image_size, get_val_transforms(image_size, imagenet_norm))
    loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

    model = build_model(model_name).to(device)
    # weights_only=False: this is our own checkpoint (contains a plain dict of
    # tensors + python scalars), not an untrusted third-party file.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_metrics = {"dice": [], "iou": [], "precision": [], "recall": [], "specificity": [], "hd95": []}
    n_images = 0
    t0 = time.time()
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            m = compute_all_metrics(logits, masks)
            for k in all_metrics:
                all_metrics[k].append(m[k])
            n_images += images.size(0)
    elapsed = time.time() - t0

    return {
        "model": model_name,
        "params": count_parameters(model),
        "ms_per_image": (elapsed / max(n_images, 1)) * 1000,
        **{k: float(np.nanmean(v)) for k, v in all_metrics.items()},
    }


def evaluate_otsu_baseline(test_pairs, image_size):
    dices, ious, precs, recs, specs, hd95s = [], [], [], [], [], []
    t0 = time.time()
    for img_path, mask_path in test_pairs:
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (image_size, image_size))
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:
            mask = mask.max(axis=2)
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        pred = otsu_baseline_predict(gray)
        dices.append(dice_coefficient(pred, mask))
        ious.append(iou_score(pred, mask))
        pr, rc, sp = precision_recall_specificity(pred, mask)
        precs.append(pr); recs.append(rc); specs.append(sp)
        hd95s.append(hausdorff_distance_95(pred, mask))
    elapsed = time.time() - t0

    return {
        "model": "classical_otsu",
        "params": 0,
        "ms_per_image": (elapsed / max(len(test_pairs), 1)) * 1000,
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "specificity": float(np.mean(specs)),
        "hd95": float(np.nanmean(hd95s)) if not np.all(np.isnan(hd95s)) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="../data/images")
    ap.add_argument("--masks_dir", default="../data/masks")
    ap.add_argument("--results_dir", default="../results")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--split_file", default=None)
    args = ap.parse_args()
    split_file = args.split_file or os.path.join(args.results_dir, "split.json")

    device = get_device()
    pairs = build_file_pairs(args.images_dir, args.masks_dir)
    with open(split_file) as f:
        idx = json.load(f)
    test_pairs = [pairs[i] for i in idx["test"]]
    print(f"Evaluating on frozen test split: {len(test_pairs)} images")

    rows = [evaluate_otsu_baseline(test_pairs, args.image_size)]

    for ckpt_path in sorted(glob.glob(os.path.join(args.results_dir, "*_best.pt"))):
        model_name = os.path.basename(ckpt_path).replace("_best.pt", "")
        print(f"Evaluating {model_name} ...")
        rows.append(evaluate_trained_model(model_name, ckpt_path, test_pairs, args.image_size, device))

    df = pd.DataFrame(rows).sort_values("dice", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(args.results_dir, "comparison_table.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== Model comparison (test set) ===")
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\nSelected model: {best['model']}  "
          f"(Dice={best['dice']:.4f}, IoU={best['iou']:.4f}, "
          f"params={int(best['params']):,}, {best['ms_per_image']:.1f} ms/image)")
    print(f"Saved comparison table -> {csv_path}")


if __name__ == "__main__":
    main()
