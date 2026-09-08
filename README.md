# Brain CT Segmentation

A small-data medical image segmentation project — training and comparing multiple deep learning architectures to segment lesions in brain CT scans, with a classical (non-deep-learning) baseline thrown in to keep myself honest.

I built this to go deep on one specific, common real-world constraint: **what do you actually do when you only have ~100 labeled medical images?** That constraint shapes almost every decision in here — the augmentation strategy, the choice to lean on transfer learning, the loss function, even which metrics I bothered to report.

## What this does

Given CT slices and their corresponding lesion masks, the pipeline:
1. Matches image/mask pairs, handles messy real-world filenames (see `src/dataset.py` — this bit off more than I expected, more below)
2. Trains 5 different segmentation architectures on an identical train/val/test split, so the comparison between them is actually fair
3. Evaluates all of them — plus a plain Otsu-thresholding baseline — on the same held-out test set
4. Spits out a single comparison table (Dice, IoU, precision, recall, HD95, params, inference time) so the "which model did we pick and why" question has a real, numbers-backed answer instead of a vibe

## Why these 5 architectures (and one non-architecture)

| Model | What it's testing |
|---|---|
| `classical_otsu` | Does deep learning even help here, or does simple thresholding get most of the way there? |
| `unet_scratch` | The baseline — vanilla U-Net, no pretrained weights |
| `unet_resnet34` | Same U-Net decoder, but with an ImageNet-pretrained encoder — isolates how much transfer learning buys you on a small dataset |
| `unetpp_resnet34` | U-Net++'s nested skip connections — worth the extra params, or does it just overfit faster on 100 images? |
| `deeplabv3plus` | Atrous convolutions + multi-scale context — useful if lesion size varies a lot slice to slice |
| `fpn_resnet34` | A lighter decoder — the "fast and cheap" point in the comparison |

All the trainable ones go through `segmentation_models_pytorch` with the exact same training loop, loss, optimizer, and data pipeline. The only thing that changes between runs is the architecture — that's the whole point of the comparison.

## The small-dataset problem, and how I dealt with it

100 images is not a lot for a model with millions of parameters. Things I leaned on:
- **Heavy but anatomically sane augmentation** — rotation, flip, brightness/contrast jitter, mild elastic deformation. Deliberately *no* vertical flips or large shears — a head CT is never upside down, and I didn't want the augmentation pipeline generating anatomically impossible training examples.
- **Transfer learning** — ImageNet-pretrained encoders where possible. Low-level filters (edges, textures) transfer surprisingly well even across the natural-image → CT domain gap, and it means the model isn't trying to learn basic visual structure from 70 training images.
- **Early stopping on validation Dice** + weight decay, because with this little data a model can start memorizing the training set within a handful of epochs.
- **A frozen train/val/test split**, shared across every architecture, so "model A beat model B" can't secretly mean "model A got an easier test split."

I'll say this upfront rather than have someone else point it out: with only ~15 images in the test set, a 1-2 point Dice difference between architectures isn't strong statistical evidence. K-fold cross-validation would be the natural next step to make that comparison more defensible — it's on my list, just not built yet.

## Why Dice/IoU and not accuracy

Lesions are a small fraction of total pixels. A model that predicts *nothing* can still score 95%+ pixel accuracy while being completely useless. That's why the loss function (BCE + Dice, with a Focal Tversky option) and the evaluation metrics are all built around overlap and boundary quality instead — see `src/losses.py` and `src/metrics.py` for the reasoning behind each one.

## The filename bug that taught me something

Worth mentioning because it's a real lesson, not a hypothetical: my first version of the image/mask matching logic assumed one extension per file (`CT_26.png` ↔ `mask_26.tif`). Real exported data broke that immediately — some files came out as `CT_83.dcm.png` (double extension from a DICOM export), and some had unrelated leading numbers (upload timestamps) before the real id. Fixed it by stripping all known extensions first, then taking the *rightmost* digit run as the id — rightmost specifically because leftmost would've silently grabbed the wrong number when the image and its mask had different timestamp prefixes but the same trailing id. It's a small function (`_extract_id` in `src/dataset.py`) but it's the kind of bug that fails silently and corrupts your whole training set if you don't catch it, so I added print warnings for any unmatched files rather than letting them fail quietly.

## Repo structure

```
brain_seg_project/
├── data/                   # your images/masks go here (not included in this repo)
├── src/
│   ├── dataset.py          # file matching + Dataset class
│   ├── augmentations.py    # train/val albumentations pipelines
│   ├── losses.py           # Dice / BCE+Dice / Focal Tversky
│   ├── metrics.py          # Dice, IoU, precision/recall/specificity, HD95
│   ├── models.py           # model zoo + classical Otsu baseline
│   ├── train.py            # trains one architecture
│   └── evaluate.py         # evaluates all checkpoints, writes comparison_table.csv
├── results/                # comparison_table.csv, per-model checkpoints & histories (generated)
├── colab_run.ipynb         # ready-to-run Colab notebook
└── run_all.sh              # trains every architecture + runs the comparison
```

## Running it

```bash
pip install -r requirements.txt

# drop your data in:
#   data/images/*.png
#   data/masks/*.tif  (any filename with a numeric id works)

bash run_all.sh
```

Or open `colab_run.ipynb` in Google Colab if you don't have a local GPU.

## Results

_(Fill this in with your `comparison_table.csv` once you've run the full sweep — this is the part that makes the comparison concrete instead of theoretical.)_

| Model | Dice | IoU | Precision | Recall | Params |
|---|---|---|---|---|---|
| classical_otsu | | | | | 0 |
| unet_scratch | | | | | |
| unet_resnet34 | | | | | |
| unetpp_resnet34 | | | | | |
| deeplabv3plus | | | | | |
| fpn_resnet34 | | | | | |

## What I'd do next with more time

- K-fold cross-validation instead of a single split, given how small the test set is
- Test-time augmentation, and possibly ensembling the top 2 models
- If the slices come from full 3D volumes, using neighboring slices (2.5D input) instead of treating each slice independently — right now that's a real limitation of this approach
