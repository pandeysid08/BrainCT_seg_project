"""
Augmentation pipelines.

Why augmentation is not optional here: with ~100 images the effective
hypothesis space a CNN can memorize is larger than the dataset, so without
augmentation you will see train Dice ~0.95 and val Dice ~0.5-0.6 (classic
overfitting gap). Augmentations are chosen to reflect realistic variation
in head CT acquisition, and deliberately EXCLUDE transforms that would
create anatomically impossible images:

- Included: small rotations (+/-15 deg, patient head tilt), horizontal flip
  (skull is roughly bilaterally symmetric), brightness/contrast jitter
  (scanner/windowing variation), slight elastic/grid distortion (soft-tissue
  deformation, small amount only), Gaussian noise (detector noise).
- Excluded: vertical flip (a scan is never upside down in practice),
  large-angle rotation or shear (anatomically implausible), heavy elastic
  distortion (would move a real bleed into an implausible shape and corrupt
  the mask's clinical meaning), color jitter (image is single-channel CT,
  not natural RGB).
- ImageNet mean/std normalization is used ONLY when the chosen encoder is
  ImageNet-pretrained (ResNet/EfficientNet backbones); the from-scratch
  U-Net instead uses simple [0,1] scaling since it has no pretrained prior
  to match.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 256, imagenet_norm: bool = True):
    norm = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) if imagenet_norm \
        else A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), translate_percent=(0.0, 0.05), scale=(0.95, 1.05), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.ElasticTransform(alpha=15, sigma=4, p=0.15),
        norm,
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 256, imagenet_norm: bool = True):
    norm = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) if imagenet_norm \
        else A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    return A.Compose([
        A.Resize(image_size, image_size),
        norm,
        ToTensorV2(),
    ])
