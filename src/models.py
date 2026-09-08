"""
Model zoo.

We compare a spread of architectures that represent genuinely different
design philosophies, not just cosmetic variants -- this is the set an
interviewer expects you to be able to justify one by one:

1. unet_scratch      - Vanilla U-Net trained from random init (Ronneberger
                        2015). The classical baseline for medical seg. No
                        pretrained prior; purely encoder-decoder with skip
                        connections. Establishes: "what does a plain
                        architecture with no pretraining or tricks achieve?"
2. unet_resnet34     - Same U-Net decoder, but encoder = ResNet-34
                        pretrained on ImageNet. Tests whether transfer
                        learning helps when you only have ~100 images (it
                        almost always does, because low-level edge/texture
                        filters transfer even across the natural-image ->
                        CT domain gap).
3. unetpp_resnet34   - U-Net++ (Zhou 2018): nested, dense skip connections
                        instead of single skip connections. Designed to
                        reduce the semantic gap between encoder and decoder
                        feature maps. Costs more params/compute; the
                        question we're answering is "is the extra
                        complexity worth it on a small dataset, or does it
                        just overfit faster?"
4. deeplabv3plus     - Atrous/dilated convolutions + ASPP (atrous spatial
                        pyramid pooling) for multi-scale context without
                        losing resolution, plus a light decoder. Good when
                        lesions vary a lot in size within the same dataset.
5. fpn_resnet34      - Feature Pyramid Network decoder: multi-scale feature
                        fusion, historically for object detection, adapted
                        for segmentation. Cheaper decoder than U-Net's,
                        useful as a lightweight/fast option in the comparison.
6. classical_otsu    - NOT a trained model at all: Otsu global thresholding
                        + morphological cleanup on the CT intensity. This
                        is the "why do we need deep learning at all" baseline
                        every segmentation project should include -- if a
                        60-year-old thresholding technique gets within a few
                        points of your CNN, that's a very different story to
                        tell in an interview than if it fails completely.

All *trainable* models are built through `segmentation_models_pytorch` so
they share the exact same training loop, loss, optimizer and data pipeline
-- the only variable across experiments is the architecture itself. This
is the single most important experimental-design point to be able to
articulate: fair comparison requires holding everything except the
independent variable constant.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(name: str, in_channels: int = 3, classes: int = 1) -> nn.Module:
    if name == "unet_scratch":
        return smp.Unet(encoder_name="resnet18", encoder_weights=None,
                         in_channels=in_channels, classes=classes)
    if name == "unet_resnet34":
        return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                         in_channels=in_channels, classes=classes)
    if name == "unetpp_resnet34":
        return smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet",
                                 in_channels=in_channels, classes=classes)
    if name == "deeplabv3plus":
        return smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights="imagenet",
                                  in_channels=in_channels, classes=classes)
    if name == "fpn_resnet34":
        return smp.FPN(encoder_name="resnet34", encoder_weights="imagenet",
                        in_channels=in_channels, classes=classes)
    raise ValueError(f"Unknown model name: {name}")


MODEL_REGISTRY = [
    "classical_otsu",     # non-trainable baseline, handled separately
    "unet_scratch",
    "unet_resnet34",
    "unetpp_resnet34",
    "deeplabv3plus",
    "fpn_resnet34",
]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def otsu_baseline_predict(gray_image: np.ndarray) -> np.ndarray:
    """Classical, non-learning baseline: Otsu threshold + small morphological
    opening/closing to remove speckle. Operates on a single-channel uint8
    CT slice and returns a binary mask (uint8, {0,1})."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(np.uint8)
