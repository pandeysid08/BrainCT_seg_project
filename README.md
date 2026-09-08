# Brain CT Segmentation

A small-data medical image segmentation project — training and comparing multiple deep learning architectures to segment lesions in brain CT scans, with a classical (non-deep-learning) baseline thrown in to keep myself honest.

I built this to go deep on one specific, common real-world constraint: **what do you actually do when you only have ~100 labeled medical images?** That constraint shapes almost every decision in here — the augmentation strategy, the choice to lean on transfer learning, the loss function, even which metrics I bothered to report.


## these 5 architectures (and one non-architecture) were used

| Model | What it's testing |
|---|---|
| `classical_otsu` | Does deep learning even help here, or does simple thresholding get most of the way there? |
| `unet_scratch` | The baseline — vanilla U-Net, no pretrained weights |
| `unet_resnet34` | Same U-Net decoder, but with an ImageNet-pretrained encoder — isolates how much transfer learning buys you on a small dataset |
| `unetpp_resnet34` | U-Net++'s nested skip connections — worth the extra params, or does it just overfit faster on 100 images? |
| `deeplabv3plus` | Atrous convolutions + multi-scale context — useful if lesion size varies a lot slice to slice |
| `fpn_resnet34` | A lighter decoder — the "fast and cheap" point in the comparison |

All the trainable ones go through `segmentation_models_pytorch` with the exact same training loop, loss, optimizer, and data pipeline. The only thing that changes between runs is the architecture — that's the whole point of the comparison.

## Results
In this format.Results file also provided for sample dataset

| Model | Dice | IoU | Precision | Recall | Params |
|---|---|---|---|---|---|
| classical_otsu | | | | | 0 |
| unet_scratch | | | | | |
| unet_resnet34 | | | | | |
| unetpp_resnet34 | | | | | |
| deeplabv3plus | | | | | |
| fpn_resnet34 | | | | | |
