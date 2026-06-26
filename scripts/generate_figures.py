#!/usr/bin/env python3
"""
Regenerate quantitative MMAP-Net manuscript figures from released CSV/prediction files.

Generated figure stems:
  fig_03_roc
  fig_04_calibration
  fig_10_per_label_diff
  fig_ablation
  fig_deferral_external_device
  fig_07_missingness
  fig_06_subgroup
  fig_domain_shift
  fig_09_seed_variability

Figures 1, 2, and 11 are included in the repository but are not regenerated here.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

LABELS = ["dr_label", "optic_disc", "macula", "quality_label"]
LABEL_NAMES = ["DR", "Optic Disc", "Macula", "Image Quality"]


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {stem}.png/.pdf")


def table04_values():
    t4 = pd.read_csv(DATA / "table04_canonical_performance.csv").set_index("label")
    return [float(t4.loc[l, "auroc"]) for l in LABELS], [float(t4.loc[l, "ece_val_fitted"]) for l in LABELS]


def fig_03_roc():
    pred = pd.read_csv(DATA / "predictions_test_seed42.csv")
    aurocs, eces = table04_values()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for lab, name, auc in zip(LABELS, LABEL_NAMES, aurocs):
        y = pred[f"targ_{lab}"].to_numpy()
        p = pred[f"prob_{lab}"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUROC={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random (AUROC=0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves for the four primary labels\n(Protocol 1, T=20, seed 42, n=3,266)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)

    ax = axes[1]
    x = np.arange(len(LABELS))
    width = 0.38
    ax.bar(x - width / 2, aurocs, width, label="AUROC")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, eces, width, alpha=0.45, label="Validation-fitted ECE")
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES, rotation=0)
    ax.set_ylim(0.86, 1.02)
    ax2.set_ylim(0, 0.04)
    ax.set_ylabel("AUROC (Protocol 1)")
    ax2.set_ylabel("Validation-fitted ECE")
    ax.set_title("AUROC and validation-fitted ECE\n(held-out test evaluation)")
    for xi, v in zip(x, aurocs):
        ax.text(xi - width / 2, v + 0.003, f"{v:.4f}", ha="center", fontsize=8)
    for xi, v in zip(x, eces):
        ax2.text(xi + width / 2, v + 0.001, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_03_roc")


def fig_04_calibration():
    test = pd.read_csv(DATA / "predictions_test_seed42.csv")
    val = pd.read_csv(DATA / "validation_probabilities_seed42.csv")
    t5 = pd.read_csv(DATA / "table05_validation_fitted_calibration.csv").set_index("label")
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    for col, (lab, name) in enumerate(zip(LABELS, LABEL_NAMES)):
        y_test = test[f"targ_{lab}"].to_numpy()
        p_test = test[f"prob_{lab}"].to_numpy()
        y_val = val[f"true_{lab}"].to_numpy()
        p_val = val[f"prob_{lab}"].to_numpy()
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_val, y_val)
        p_cal = iso.transform(p_test)
        raw_ece = float(t5.loc[lab, "raw_ece"])
        cal_ece = float(t5.loc[lab, "validation_fitted_ece"])

        for row, (probs, ece, ttl) in enumerate([
            (p_test, raw_ece, "Before calibration"),
            (p_cal, cal_ece, "After validation-fitted isotonic regression"),
        ]):
            ax = axes[row, col]
            frac, mean = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
            ax.plot(mean, frac, marker="o", linewidth=1.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{name}\n{ttl}\n(ECE={ece:.3f})", fontsize=8)
            if col == 0:
                ax.set_ylabel("Observed frequency")
            if row == 1:
                ax.set_xlabel("Mean predicted probability")
            ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, "fig_04_calibration")


def fig_ablation():
    t6 = pd.read_csv(DATA / "table06_ablation.csv")
    labels = ["Image-only", "Metadata-only", "Early fusion", "no_missingness", "no_gate", "MMAP-Net"]
    keys = ["image_only", "meta_only", "early_fusion", "mmap_no_missingness", "mmap_no_gate", "mmap_full"]
    df = t6.set_index("model").loc[keys]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, col, title, ylim in [
        (axes[0], "macro_auroc", "Macro-AUROC (Protocol 2)", (0.60, 0.95)),
        (axes[1], "macro_f1", "Macro-F1 after threshold tuning", (0.45, 0.85)),
    ]:
        vals = df[col].astype(float).to_numpy()
        bars = ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f"{v:.4f}", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_ablation")


def fig_10_per_label_diff():
    t7 = pd.read_csv(DATA / "table07_per_label_diff.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, comp, title in [
        (axes[0], "vs_image_only", "MMAP-Net Full vs. Image-only"),
        (axes[1], "vs_no_gate", "MMAP-Net Full vs. no_gate"),
    ]:
        sub = t7[t7["comparison"] == comp].set_index("label").loc[LABELS]
        vals = sub["auroc_diff"].astype(float).to_numpy()
        y = np.arange(len(vals))
        ax.barh(y, vals)
        ax.axvline(0, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(LABEL_NAMES)
        ax.set_xlabel("AUROC Difference (MMAP-Net - Baseline)")
        ax.set_title(title)
        for yi, v in zip(y, vals):
            ax.text(v + (0.002 if v >= 0 else -0.002), yi, f"{v:+.4f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_10_per_label_diff")


def fig_deferral_external_device():
    t8 = pd.read_csv(DATA / "table08_deferral_external_device.csv")
    x = t8["coverage_target"].astype(str).to_numpy()
    br = t8["brset_deferral_pct"].astype(float).to_numpy()
    mb = t8["mbrset_deferral_pct"].astype(float).to_numpy()
    diff = t8["mbrset_minus_brset_pp"].astype(float).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(x, br, marker="o", label="BRSET test")
    ax.plot(x, mb, marker="s", linestyle="--", label="mBRSET external-device")
    ax.set_xlabel("Validation Target (%)")
    ax.set_ylabel("Deferral Rate (%)")
    ax.set_title("Score-calibrated deferral rate\nBRSET vs mBRSET")
    ax.legend()
    ax.grid(alpha=0.25)
    for xi, v in zip(x, br): ax.text(xi, v + 2, f"{v:.1f}%", ha="center", fontsize=8)
    for xi, v in zip(x, mb): ax.text(xi, v + 2, f"{v:.1f}%", ha="center", fontsize=8)

    ax = axes[1]
    bars = ax.bar(x, diff)
    ax.set_xlabel("Validation Target")
    ax.set_ylabel("mBRSET - BRSET deferral difference (pp)")
    ax.set_title("External-device deferral-rate difference")
    ax.grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, diff):
        ax.text(bar.get_x()+bar.get_width()/2, v+1, f"+{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_deferral_external_device")


def fig_07_missingness():
    t9 = pd.read_csv(DATA / "table09_missingness_robustness.csv").set_index("label").loc[LABELS]
    x = np.arange(len(LABELS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.bar(x - 0.2, t9["auroc_full_meta"].astype(float), 0.4, label="MMAP-Net full metadata")
    ax.bar(x + 0.2, t9["auroc_30pct_mcar"].astype(float), 0.4, label="MMAP-Net 30% MCAR")
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES, rotation=20, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title("Controlled 30% MCAR metadata masking")
    ax.set_ylim(0.88, 0.98)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    width = 0.35
    ax.bar(x - width/2, t9["auroc_drop_mmap"].astype(float), width, label="MMAP-Net")
    ax.bar(x + width/2, t9["auroc_drop_ef"].astype(float), width, label="Early fusion")
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES, rotation=20, ha="right")
    ax.set_ylabel("AUROC change")
    ax.set_title("Signed AUROC change")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_07_missingness")


def fig_06_subgroup():
    t11 = pd.read_csv(DATA / "table11_subgroup_consistency.csv")
    x = np.arange(len(t11))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    cols = [("auroc", "DR AUROC", (0.996, 1.000)), ("ece", "Raw ECE", None), ("coverage90_pct", "Coverage@90 (%)", (98.8, 100.1))]
    for ax, (col, title, ylim) in zip(axes, cols):
        vals = t11[col].astype(float).to_numpy()
        bars = ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(t11["subgroup"], rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        if ylim: ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        for bar, v in zip(bars, vals):
            fmt = f"{v:.4f}" if col == "auroc" else f"{v:.3f}" if col == "ece" else f"{v:.2f}"
            ax.text(bar.get_x()+bar.get_width()/2, v, fmt, ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_06_subgroup")


def fig_domain_shift():
    t13 = pd.read_csv(DATA / "table13_external_device_check.csv")
    labels = ["DR", "Image Quality", "Hypertension (aux.)"]
    x = np.arange(len(labels))
    br = t13["brset_auroc"].astype(float).to_numpy()
    mb = t13["mbrset_auroc"].astype(float).to_numpy()
    decline = t13["absolute_decline"].astype(float).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    width = 0.35
    ax.bar(x - width/2, br, width, label="BRSET test")
    ax.bar(x + width/2, mb, width, label="mBRSET external-device")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.50, 1.05)
    ax.set_title("BRSET vs mBRSET")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    for xi, v in zip(x - width/2, br): ax.text(xi, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)
    for xi, v in zip(x + width/2, mb): ax.text(xi, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)
    ax = axes[1]
    bars = ax.bar(x, decline)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute AUROC decline")
    ax.set_title("BRSET - mBRSET AUROC decline")
    ax.set_ylim(0, 0.22)
    ax.grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, decline):
        ax.text(bar.get_x()+bar.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_domain_shift")


def fig_09_seed_variability():
    df = pd.read_csv(DATA / "figure12_seed_reproducibility.csv")
    rows = df[df["seed"].isin(["42", "123", "777"])]
    if rows.empty:
        rows = df.iloc[:3]
    x = np.arange(len(rows))
    vals = rows["macro_auroc"].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in rows["seed"]])
    ax.set_ylabel("Macro-AUROC (Protocol 2)")
    ax.set_xlabel("Seed")
    ax.set_ylim(0.922, 0.934)
    ax.set_title("Three-seed reproducibility")
    ax.grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v + 0.0003, f"{v:.4f}", ha="center", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_09_seed_variability")


def main():
    fig_03_roc()
    fig_04_calibration()
    fig_10_per_label_diff()
    fig_ablation()
    fig_deferral_external_device()
    fig_07_missingness()
    fig_06_subgroup()
    fig_domain_shift()
    fig_09_seed_variability()


if __name__ == "__main__":
    main()
