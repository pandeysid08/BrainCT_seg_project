#!/usr/bin/env bash
# Trains every architecture in the comparison with identical settings
# (same seed, same frozen split, same loss/optimizer/schedule), then
# runs the head-to-head evaluation against the classical baseline.
#
# Usage: bash run_all.sh
set -e
cd "$(dirname "$0")/src"

MODELS=("unet_scratch" "unet_resnet34" "unetpp_resnet34" "deeplabv3plus" "fpn_resnet34")
EPOCHS=100
BATCH_SIZE=8
LOSS=bce_dice

for m in "${MODELS[@]}"; do
  echo "=================================================="
  echo " Training: $m"
  echo "=================================================="
  python3 train.py --model "$m" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --loss "$LOSS"
done

echo "=================================================="
echo " Evaluating all models on the frozen test split"
echo "=================================================="
python3 evaluate.py
