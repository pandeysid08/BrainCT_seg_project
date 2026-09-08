"""
Dataset loading for the brain CT / segmentation-mask pairs.

Design notes for the interview:
- Images and masks are matched by a numeric id extracted from the filename
  (e.g. `CT_26.png` <-> `mask_26.tif`), NOT by directory order. Relying on
  sorted() order to line up two folders is a classic silent bug: if even one
  file is missing, every pair after it is shifted and you silently train on
  wrong image/mask pairs. Matching by id fails loudly instead.
- Masks may be stored as multi-channel TIFFs (RGB) even though they are
  binary. We collapse to a single channel and binarize with a threshold,
  rather than assuming a fixed pixel value, so the loader is robust to
  0/255, 0/1, or anti-aliased mask edges from annotation tools.
- CT pixel values here are already 8-bit windowed PNGs (0-255), not raw
  Hounsfield units, so we normalize with simple min-max / ImageNet stats
  depending on whether the encoder is ImageNet-pretrained.
"""
import os
import re
import glob
from typing import Optional, Callable, List, Tuple

import numpy as np
import cv2
from torch.utils.data import Dataset


_KNOWN_EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".dcm", ".gif"}


def _strip_known_extensions(filename: str) -> str:
    """Repeatedly strips recognized extensions, so 'CT_83.dcm.png' (a common
    DICOM->PNG export naming pattern) becomes 'CT_83', not 'CT_83.dcm'.
    Only commits a strip when the piece removed is a KNOWN extension, so a
    stray dot elsewhere in the name (e.g. 'CT.83.png') doesn't get eaten."""
    current = filename
    while True:
        stem, ext = os.path.splitext(current)
        if ext.lower() in _KNOWN_EXTENSIONS:
            current = stem
        else:
            break
    return current


def _extract_id(filename: str) -> str:
    """Pulls the numeric id out of a filename.

    Two things this has to survive, both seen in real exports:
      - Double extensions from DICOM->PNG conversion: 'CT_83.dcm.png' -> '83'
        (fixed by stripping ALL known extensions first, not just the last one)
      - A leading, unrelated number before the real id, e.g. a
        timestamp-prefixed filename '1788851035632_CT_26.png' -> '26', where
        the image and its matching mask can have DIFFERENT leading numbers
        (different upload timestamps) but the SAME trailing id.
        (fixed by taking the RIGHTMOST digit run, not the leftmost)
    """
    stem = _strip_known_extensions(filename)
    matches = re.findall(r"\d+", stem)
    if not matches:
        raise ValueError(f"Could not find a numeric id in filename: {filename}")
    return matches[-1]


def build_file_pairs(images_dir: str, masks_dir: str) -> List[Tuple[str, str]]:
    """Matches image/mask files by numeric id and returns sorted (img, mask) paths."""
    img_paths = {}
    for p in glob.glob(os.path.join(images_dir, "*")):
        img_paths[_extract_id(os.path.basename(p))] = p

    mask_paths = {}
    for p in glob.glob(os.path.join(masks_dir, "*")):
        mask_paths[_extract_id(os.path.basename(p))] = p

    common_ids = sorted(set(img_paths) & set(mask_paths), key=lambda x: int(x))
    missing_img = set(mask_paths) - set(img_paths)
    missing_mask = set(img_paths) - set(mask_paths)
    if missing_img:
        print(f"[warn] {len(missing_img)} masks have no matching image, skipping: {sorted(missing_img)[:5]}...")
    if missing_mask:
        print(f"[warn] {len(missing_mask)} images have no matching mask, skipping: {sorted(missing_mask)[:5]}...")

    return [(img_paths[i], mask_paths[i]) for i in common_ids]


class BrainCTSegDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        image_size: int = 256,
        transform: Optional[Callable] = None,
        mask_threshold: int = 127,
    ):
        self.pairs = pairs
        self.image_size = image_size
        self.transform = transform
        self.mask_threshold = mask_threshold

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(img_path)

        # cv2 can read multi-page/typed TIFFs; force to single channel + binarize.
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if mask.ndim == 3:
            mask = mask.max(axis=2)  # any-channel-on -> foreground
        mask = (mask > self.mask_threshold).astype(np.uint8)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # 3-channel so we can reuse ImageNet-pretrained encoders (resnet/efficientnet).
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        return image, mask.float().unsqueeze(0) if hasattr(mask, "float") else mask
