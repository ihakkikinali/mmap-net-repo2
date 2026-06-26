# Parameter count per model variant (no training; numel only).
# Model definitions (e.g. MMAPNetBase) must be loaded from the notebook first.
import torch

# NOTE: This is a TEMPLATE. build_model must be supplied from the original
# training notebook (model definitions are not redistributed in this review
# package). The verified parameter count for the full model is 69,369,413
# (loads with strict=True); all counts are in ../data/table15_complexity.csv.


def count(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

# Build each variant and count. Skeleton — adapt to your model builder:
configs = {
    'image_only':          dict(use_meta=False, use_gate=False),
    'metadata_only':       dict(use_image=False),
    'early_fusion':        dict(fusion='concat', use_gate=False),
    'mmap_no_missingness': dict(use_indicator=False),
    'mmap_no_gate':        dict(use_gate=False),
    'mmap_full':           dict(),  # full model
}

print(f"{'Model':24s} {'Total (M)':>10s} {'Trainable (M)':>14s}")
for name, cfg in configs.items():
    # model = build_model(**cfg)   # provide model builder from the original notebook
    raise SystemExit(
        "This is a documented template. To run it, load the original model "
        "definitions and provide build_model from the training notebook. "
        "Verified counts are in ../data/table15_complexity.csv."
    )
    t, tr = count(model)
    print(f"{name:24s} {t/1e6:10.2f} {tr/1e6:14.2f}")

# Optional GFLOPs estimation:
# from thop import profile
# x_img = torch.randn(1,3,256,256); x_meta = torch.randn(1, META_DIM)
# flops, _ = profile(model, inputs=(x_img, x_meta))
# print(f"GFLOPs: {flops/1e9:.1f}")
