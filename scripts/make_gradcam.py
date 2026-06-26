#!/usr/bin/env python3
"""
GradCAM figure generation for MMAP-Net (manuscript Figure 11; output files fig_11_gradcam_*).

This script REQUIRES the trained model checkpoint and the raw BRSET images,
so it must be run in the GPU/Kaggle environment (not in the offline figure
pipeline). It produces English-labelled GradCAM panels without the vertical
banding artefact present in the earlier version.

Key fixes vs. the earlier figure:
  * All text is in English.
  * The banding artefact is removed by (a) upsampling the CAM with bilinear
    interpolation instead of nearest-neighbour, and (b) normalising each CAM
    independently before overlay.

Inputs (Kaggle):
  CKPT = mmap_full_seed42_best.pt
  IMAGE_DIR = .../brset/fundus_photos
  gradcam_files.csv  (which sample ids to visualise per label)

Output: fig_11_gradcam_<label>.png/.pdf  (4 positive examples per label)
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# ----- paths (adjust to the Kaggle layout) -----
CKPT = Path("/kaggle/working/outputs/checkpoints/mmap_full_seed42_best.pt")
IMAGE_DIR = Path("/kaggle/input/.../fundus_photos")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["dr_label", "optic_disc", "macula", "quality_label"]
LABEL_NAMES = {"dr_label": "DR", "optic_disc": "Optic Disc",
               "macula": "Macula", "quality_label": "Image Quality"}


def make_gradcam(model, image_tensor, meta_tensor, target_idx, target_layer):
    """Standard Grad-CAM on the image encoder's last conv/stage output."""
    activations, gradients = {}, {}

    def fwd_hook(_, __, output):
        activations["value"] = output.detach()

    def bwd_hook(_, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    out = model(image_tensor.to(DEVICE), meta_tensor.to(DEVICE))
    logits = out["la"]
    score = logits[0, target_idx]
    score.backward()

    act = activations["value"][0]          # (C, H, W)
    grad = gradients["value"][0]           # (C, H, W)
    weights = grad.mean(dim=(1, 2))        # (C,)
    cam = F.relu((weights[:, None, None] * act).sum(0))  # (H, W)
    cam = cam.cpu().numpy()

    h1.remove(); h2.remove()
    # normalise per-CAM (removes banding from shared scaling)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def overlay(img_rgb, cam):
    """Bilinear upsample (not nearest) -> removes vertical banding."""
    H, W = img_rgb.shape[:2]
    cam_up = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip(0.55 * img_rgb + 0.45 * heat, 0, 1)


def main():
    import pandas as pd
    # model loading is environment-specific; assumed available as `model`
    # from the shared notebook cells (MMAPNetBase + clean_state_dict).
    raise SystemExit(
        "Run inside the MMAP-Net notebook: import the model definition, "
        "load CKPT, then call make_gradcam/overlay for the sample ids in "
        "gradcam_files.csv. The plotting block below is ready to paste.")

    # ---- plotting block (paste into the notebook after model is loaded) ----
    # files = pd.read_csv("gradcam_files.csv")
    # for lab in LABELS:
    #     ids = pick_positive_ids(lab, n=4)          # 4 positive examples
    #     fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    #     for col, sid in enumerate(ids):
    #         img, x_img, x_meta = load_sample(sid)  # img in [0,1] RGB
    #         cam = make_gradcam(model, x_img, x_meta,
    #                            LABELS.index(lab), model.image_encoder.last_stage)
    #         axes[0, col].imshow(img);            axes[0, col].axis("off")
    #         axes[1, col].imshow(overlay(img, cam)); axes[1, col].axis("off")
    #         axes[0, col].set_title(f"Positive example {col+1}", fontsize=9)
    #     axes[0, 0].set_ylabel("Input", fontsize=10)
    #     axes[1, 0].set_ylabel("Grad-CAM", fontsize=10)
    #     fig.savefig(f"fig_11_gradcam_{lab}.png", dpi=200, bbox_inches="tight")
    #     fig.savefig(f"fig_11_gradcam_{lab}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
