# Brain CT Segmentation — Project Guide & Interview Notes

> Run in Colab: open `colab_run.ipynb` in this repo via **File -> Open notebook -> GitHub** in Colab, or upload it directly (see "Running in Colab" section below).

This project trains and compares several segmentation architectures on a
small (~100-image) brain CT dataset with binary masks (e.g. hemorrhage /
lesion region vs. background), and picks a final model using a
metrics-driven comparison rather than a single number.

Verified: the code in this repo was smoke-tested end to end (data loading →
augmentation → training loop → checkpointing → evaluation → comparison
table) before being handed to you. The one thing *not* exercised here is
downloading ImageNet-pretrained encoder weights, because this sandbox
blocks that specific network call — on your own machine it will download
automatically on first run.

---

## 1. Problem framing

- **Task type:** binary semantic segmentation (pixel-wise classification:
  lesion vs. background), not classification or detection. This distinction
  matters because it drives every downstream choice: loss function
  (pixel-overlap losses, not cross-entropy-only), metrics (Dice/IoU, not
  accuracy), and architecture family (encoder-decoder with skip
  connections, not a plain CNN classifier).
- **Input:** grayscale head CT slices (8-bit windowed PNG, 512x512 in your
  sample).
- **Output:** binary mask, same spatial size as input, 1 = lesion.
- **Data scale:** ~100 images is a *small-data* medical imaging problem.
  This single fact should shape almost every design decision you're asked
  to defend in an interview: heavy augmentation, transfer learning,
  aggressive regularization, k-fold cross-validation as a serious option,
  and skepticism toward very large/complex architectures.

---

## 2. Data pipeline (`src/dataset.py`, `src/augmentations.py`)

**Matching image/mask pairs by numeric ID, not by folder order.** Two
folders sorted independently and zipped together will silently misalign if
even one file is missing — you'd train against wrong masks with no error
raised. Matching by the numeric id extracted from the filename fails loudly
instead (prints a warning and drops unmatched files).

**Mask normalization.** Your uploaded mask is an RGB-encoded TIFF even
though it's logically binary (values are 0/255 across all channels). The
loader collapses to single channel and thresholds, rather than assuming a
fixed encoding — this makes it robust to whatever export format your
annotation tool produced (0/1, 0/255, or anti-aliased edges).

**Train/val/test split — 70/15/15, frozen once.** With so few images, the
split is written to `results/split.json` the first time you run training,
and every subsequent architecture reuses the *exact same* split. This is
non-negotiable for a fair comparison: if each model saw a different random
test set, "model A beat model B by 2 Dice points" could just mean "model A
got an easier test split."

**Augmentation is the main defense against overfitting on ~100 images**
(README section 3 below). Included: horizontal flip, small
rotation/translation/scale, brightness/contrast jitter, mild elastic
deformation, Gaussian noise. Excluded on purpose: vertical flip (a head CT
is never upside down), large rotations/shear (anatomically implausible),
heavy elastic distortion (would move a real lesion into a shape that no
longer matches its ground-truth mask), color jitter (this is single-channel
CT, not natural color photography — jittering hue/saturation is meaningless
and can only hurt).

---

## 3. Why 100 images changes everything

This is the question most likely to come up, so it's worth having a crisp
answer:

1. **Overfitting risk is high.** A modern CNN has millions of parameters;
   100 images is nowhere near enough to constrain that many degrees of
   freedom from scratch. Mitigations used here: augmentation, weight decay,
   early stopping on validation Dice, dropout implicit in some encoders,
   and preferring smaller/pretrained encoders over huge from-scratch nets.
2. **Transfer learning matters a lot.** Encoders pretrained on ImageNet
   already know general-purpose edge/texture/shape filters. Even though
   natural photos and CT slices look very different, low-level filters
   transfer surprisingly well, and it means the model isn't learning basic
   visual structure from 70 training images — it's fine-tuning. This is
   why `unet_resnet34` is expected to beat `unet_scratch`, and it's the
   single most standard technique for making small medical-imaging
   datasets tractable.
3. **Cross-validation is worth considering over a single split.** A single
   70/15/15 split of 100 images means the validation set is ~15 images —
   noisy. If you want a more defensible number for an interview, k-fold
   (e.g. 5-fold) cross-validation and reporting mean ± std Dice is more
   convincing than a single train/val/test run. This repo ships the
   simpler single-split version for clarity; the natural follow-up
   engineering task is wrapping `train.py`'s split logic in a k-fold loop.
4. **Simpler/classical baselines become genuinely competitive.** With so
   little data, a well-tuned classical method (Otsu thresholding here) can
   sometimes come close to a small CNN. That's *why the classical baseline
   is in the comparison at all* — it's the "do we even need deep learning"
   check every small-data project should run before reaching for a
   14M-parameter network.
5. **Statistical significance of "model A > model B" is weak on 15 test
   images.** Be ready to say this out loud rather than overclaim: a 1-2
   point Dice difference between two architectures on a 15-image test set
   is not strong evidence of a real difference. Larger test sets or
   cross-validated results would be needed to be confident.

---

## 4. Models compared (`src/models.py`)

| Model | Idea | Why it's in the comparison |
|---|---|---|
| `classical_otsu` | Otsu global thresholding + morphological cleanup, **no training** | The "do we need deep learning at all" baseline |
| `unet_scratch` | Classic U-Net (Ronneberger 2015), random init, ResNet-18-shaped encoder | Baseline architecture, no transfer learning |
| `unet_resnet34` | Same U-Net decoder, **ImageNet-pretrained** ResNet-34 encoder | Isolates the effect of transfer learning |
| `unetpp_resnet34` | U-Net++ (Zhou 2018): nested/dense skip connections | Tests whether reducing the encoder-decoder semantic gap helps, at the cost of more parameters |
| `deeplabv3plus` | Atrous convolutions + ASPP (multi-scale context) + light decoder | Tests whether explicit multi-scale context helps if lesion size varies |
| `fpn_resnet34` | Feature Pyramid Network decoder | Cheaper decoder — a fast/lightweight point in the comparison |

All five trainable models are built through `segmentation_models_pytorch`
so they share **identical** training loop, loss, optimizer, LR schedule,
and data pipeline. This is the key experimental-design principle: to claim
an architecture is genuinely better, everything else has to be held
constant. Only the architecture is the independent variable.

---

## 5. Loss functions (`src/losses.py`)

The central issue: a lesion is often a small fraction of total pixels
(class imbalance). Plain pixel accuracy or plain BCE is dominated by the
easy background class — a model predicting nothing at all can still score
95%+ "accuracy."

- **Dice loss** — directly optimizes pixel overlap; robust to imbalance but
  has unstable/near-zero gradients very early in training when the model
  predicts nothing on the foreground yet.
- **BCE + Dice (default, `bce_dice`)** — BCE supplies a stable per-pixel
  gradient from step one; Dice supplies the imbalance-robust overlap
  signal. This combination is the most common default in medical image
  segmentation and is what `run_all.sh` uses.
- **Focal Tversky loss** — generalizes Dice with separate false-positive
  (α) vs false-negative (β) weights. Set β > α to penalize missed lesion
  pixels harder than spurious ones — relevant if the clinical cost of
  missing part of a bleed is higher than a slightly over-drawn mask.

Be ready to explain: **why not just cross-entropy?** → because it treats
every pixel independently and equally, so with 95%+ background pixels the
optimal-looking solution (in terms of loss) is close to predicting all
background.

---

## 6. Metrics (`src/metrics.py`) — and why each one is there

| Metric | Catches |
|---|---|
| **Dice** | Standard medical segmentation metric; overlap-based, forgiving of exact boundary but sensitive to small structures |
| **IoU / Jaccard** | Related to Dice (`IoU = Dice / (2 - Dice)`) but stricter; reported because different papers standardize on one or the other |
| **Precision** | Over-segmentation / false alarms |
| **Recall (sensitivity)** | Missed lesion pixels — usually the clinically worse failure mode |
| **Specificity** | Included specifically to show why it's a *bad* headline metric here — with a huge background class it stays near 1.0 for both a good model and a broken one |
| **HD95 (95th-percentile Hausdorff distance)** | Boundary-level metric; catches cases where two masks have similar Dice but one has a stray blob far from the true lesion, which Dice alone won't flag. Computed pixel-wise, undefined (NaN) if either mask is empty |
| **Params / ms-per-image** | The accuracy-vs-cost trade-off — a 1% Dice gain that costs 5x inference latency is a real trade-off, not a free win |

**Why not just report accuracy?** Because pixel accuracy is dominated by
the background class in a segmentation task with heavy class imbalance —
it can look excellent while the model is clinically useless. This is one
of the most common "gotcha" interview questions in segmentation projects.

---

## 7. Training details (`src/train.py`)

- **Optimizer:** AdamW, small weight decay (1e-4) as a regularizer.
- **LR schedule:** `ReduceLROnPlateau` on validation Dice — adapts to
  actual training dynamics rather than committing to a fixed schedule in
  advance, which is hard to tune well with so few iterations/epoch.
- **Batch size:** kept small (4–8) — the dataset is tiny, and a large batch
  means very few gradient updates per epoch, plus noisier BatchNorm
  statistics in the encoder at very small batch counts. (If BatchNorm
  proves unstable at your chosen batch size, GroupNorm is a documented
  alternative worth raising in discussion.)
- **Early stopping:** on validation Dice, patience configurable — the main
  defense against a model memorizing 70 training images.
- **Reproducibility:** every RNG source seeded (`src/utils.py`), deterministic
  cuDNN kernels. Every architecture also uses the identical frozen split.

---

## 8. How to run this yourself

```bash
pip install -r requirements.txt

# 1. Put your data here (any filenames containing a numeric id, e.g. CT_26.png / mask_26.tif):
#    data/images/*.png
#    data/masks/*.tif

# 2. Train every architecture + run the head-to-head comparison:
bash run_all.sh

# ...or run/inspect one model at a time:
cd src
python3 train.py --model unet_resnet34 --epochs 100 --loss bce_dice
python3 evaluate.py
```

Outputs land in `results/`:
- `split.json` — the frozen train/val/test split (shared by every model)
- `<model>_best.pt` — best checkpoint per architecture (by val Dice)
- `<model>_history.json` — per-epoch loss/Dice/IoU curves
- `comparison_table.csv` — the final head-to-head table across every model + the classical baseline, sorted by test Dice

---

## 9. How to defend "why we picked this model"

`evaluate.py` doesn't just print numbers — it sorts by Dice and states the
selection at the bottom. The story to tell in an interview is never just
"highest Dice wins." Walk through, in order:

1. **Primary metric:** test-set Dice (and IoU, which moves with it) —
   the standard segmentation metric, computed on data none of the models
   were fit or tuned against.
2. **Boundary quality (HD95):** if two models are close on Dice, HD95
   differentiates "a clean, tight boundary" from "the right area but a
   ragged/outlying boundary."
3. **Failure mode balance (precision vs recall):** decide, given the
   clinical framing, whether missed lesion pixels (recall) or false
   alarms (precision) matter more, and let that break ties.
4. **Cost:** parameter count and ms/image — is the accuracy gain worth the
   extra compute for how this would actually be deployed?
5. **Honesty about sample size:** with a ~15-image test set, note that
   small Dice differences between architectures are not strongly
   statistically significant, and that k-fold cross-validation would be
   the natural next step to firm this conclusion up.

---

## 10. Likely interview questions (and where the answer lives in this repo)

- *"Why Dice/IoU and not accuracy?"* → Section 6.
- *"How do you handle class imbalance?"* → Section 5 (loss), Section 6
  (metrics that don't hide it).
- *"How do you prevent overfitting with only 100 images?"* → Section 3
  (augmentation, transfer learning, early stopping, weight decay).
- *"Why compare against a non-deep-learning baseline?"* → Section 4,
  `classical_otsu` row.
- *"How did you make sure the comparison was fair?"* → Section 4 (shared
  training loop/loss/optimizer), Section 2 (frozen split).
- *"What would you do differently with more data / more time?"* → k-fold
  CV (Section 3, point 3), test-time augmentation, ensembling top-2
  models, 3D context (if slices come from full volumes, a 2.5D or 3D
  model could use neighboring slices — this project treats each slice
  independently, which is a real limitation worth naming).
- *"What's a limitation of Dice as a metric?"* → It's insensitive to
  small structures' exact boundary quality and can look similar for very
  different failure modes — this is exactly why HD95 and
  precision/recall are reported alongside it, not instead of it.

---

## 11. Repo structure

```
brain_seg_project/
├── data/
│   ├── images/          # put your CT PNGs here
│   └── masks/            # put your mask TIFFs/PNGs here
├── src/
│   ├── dataset.py         # file matching + Dataset class
│   ├── augmentations.py   # train/val albumentations pipelines
│   ├── losses.py          # Dice / BCE+Dice / Focal Tversky
│   ├── metrics.py         # Dice, IoU, precision/recall/specificity, HD95
│   ├── models.py          # model zoo + classical Otsu baseline
│   ├── train.py           # trains one architecture, saves best checkpoint
│   ├── evaluate.py        # evaluates all checkpoints + baseline, writes comparison_table.csv
│   └── utils.py           # seeding, AverageMeter, EarlyStopping
├── results/                # split.json, checkpoints, histories, comparison_table.csv (generated)
├── run_all.sh              # trains every architecture + runs evaluate.py
└── requirements.txt
```
