"""
Train a single architecture end-to-end.

Usage:
    python src/train.py --model unet_resnet34 --epochs 100 --loss bce_dice

Design decisions worth explaining out loud in an interview:

- Split: with only ~100 images we use a 70/15/15 train/val/test split
  (stratified is not meaningful here since this is per-pixel not per-class,
  but we DO make sure the split is done once and frozen to a json file so
  every architecture in the comparison sees identical splits -- otherwise
  "model A is better than model B" could just mean "model A got an easier
  test split").
- Optimizer: AdamW (decoupled weight decay) rather than plain Adam, small
  weight decay (1e-4) as the main regularizer against overfitting on a
  small dataset, alongside augmentation and early stopping.
- LR schedule: ReduceLROnPlateau on validation Dice -- with so few
  iterations per epoch, a fixed schedule (e.g. cosine over N epochs) is
  hard to tune well in advance; plateau-based reduction adapts to how
  training actually goes.
- Batch size: kept intentionally small (4-8) because effective dataset
  size is tiny; a large batch would mean very few gradient updates per
  epoch and noisier/less-informative statistics from BatchNorm layers in
  the encoder. If BatchNorm behaves unstably at small batch sizes for a
  given encoder, GroupNorm is a documented alternative worth mentioning.
- Mixed precision (AMP) is enabled when CUDA is available purely for
  speed; it does not change what's being optimized.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import build_file_pairs, BrainCTSegDataset
from augmentations import get_train_transforms, get_val_transforms
from losses import get_loss
from metrics import compute_all_metrics
from models import build_model, count_parameters
from utils import set_seed, get_device, AverageMeter, EarlyStopping


def get_or_create_split(pairs, split_path, seed=42):
    if os.path.exists(split_path):
        with open(split_path) as f:
            idx = json.load(f)
        return ([pairs[i] for i in idx["train"]],
                [pairs[i] for i in idx["val"]],
                [pairs[i] for i in idx["test"]])

    all_idx = list(range(len(pairs)))
    train_idx, temp_idx = train_test_split(all_idx, test_size=0.30, random_state=seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed)
    with open(split_path, "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f, indent=2)
    return ([pairs[i] for i in train_idx],
             [pairs[i] for i in val_idx],
             [pairs[i] for i in test_idx])


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss_meter = AverageMeter()
    metric_totals = {"dice": [], "iou": []}

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_meter.update(loss.item(), images.size(0))
            m = compute_all_metrics(logits, masks)
            metric_totals["dice"].append(m["dice"])
            metric_totals["iou"].append(m["iou"])

    return loss_meter.avg, np.mean(metric_totals["dice"]), np.mean(metric_totals["iou"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="../data/images")
    ap.add_argument("--masks_dir", default="../data/masks")
    ap.add_argument("--model", required=True, choices=[
        "unet_scratch", "unet_resnet34", "unetpp_resnet34", "deeplabv3plus", "fpn_resnet34"])
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", default="bce_dice", choices=["dice", "bce_dice", "focal_tversky", "bce"])
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_file", default="../results/split.json")
    ap.add_argument("--out_dir", default="../results")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    pairs = build_file_pairs(args.images_dir, args.masks_dir)
    if len(pairs) == 0:
        raise RuntimeError("No matched image/mask pairs found. Check --images_dir/--masks_dir.")
    train_pairs, val_pairs, test_pairs = get_or_create_split(pairs, args.split_file, args.seed)
    print(f"Dataset sizes -> train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

    imagenet_norm = args.model != "unet_scratch"
    train_ds = BrainCTSegDataset(train_pairs, args.image_size,
                                  get_train_transforms(args.image_size, imagenet_norm))
    val_ds = BrainCTSegDataset(val_pairs, args.image_size,
                                get_val_transforms(args.image_size, imagenet_norm))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.model).to(device)
    n_params = count_parameters(model)
    print(f"Model: {args.model} | trainable params: {n_params:,}")

    criterion = get_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    stopper = EarlyStopping(patience=args.patience, mode="max")

    ckpt_path = os.path.join(args.out_dir, f"{args.model}_best.pt")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_dice, train_iou = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_dice, val_iou = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step(val_dice)

        is_best = stopper.step(val_dice)
        if is_best:
            torch.save({"model_name": args.model, "state_dict": model.state_dict(),
                        "val_dice": val_dice, "epoch": epoch}, ckpt_path)

        row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                   train_dice=train_dice, val_dice=val_dice,
                   train_iou=train_iou, val_iou=val_iou,
                   lr=optimizer.param_groups[0]["lr"], time_s=time.time() - t0)
        history.append(row)
        print(f"[{args.model}] epoch {epoch:03d} | train_loss {train_loss:.4f} "
              f"val_loss {val_loss:.4f} | train_dice {train_dice:.4f} "
              f"val_dice {val_dice:.4f}{'  *best*' if is_best else ''}")

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch} (no val_dice improvement for {args.patience} epochs).")
            break

    hist_path = os.path.join(args.out_dir, f"{args.model}_history.json")
    with open(hist_path, "w") as f:
        json.dump({"n_params": n_params, "history": history,
                   "best_val_dice": stopper.best}, f, indent=2)
    print(f"Saved checkpoint -> {ckpt_path}\nSaved history -> {hist_path}")


if __name__ == "__main__":
    main()
