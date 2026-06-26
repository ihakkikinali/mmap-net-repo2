#!/usr/bin/env python3
"""
Verify released MMAP-Net result artifacts against manuscript tables.

Checks performed:
  1. Recompute Protocol 2 macro-AUROC for released per-sample caches
     (`mmap_full`, `mmap_no_missingness`) and compare with Table 6.
  2. Recompute held-out test AUROC values from the released test prediction
     export and compare with Table 4.
  3. Check arithmetic consistency for selected table deltas.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = ["dr_label", "optic_disc", "macula", "quality_label"]


def macro_auroc(probs, targets):
    return float(np.mean([roc_auc_score(targets[:, i], probs[:, i]) for i in range(probs.shape[1])]))


def close(a, b, tol=1e-3):
    return abs(float(a) - float(b)) <= tol


def main():
    ok = True
    print("=" * 72)
    print("MMAP-Net released-artifact consistency check")
    print("=" * 72)

    print("\n[1] Released per-sample caches vs Table 6 (Protocol 2, T=1)")
    with open(DATA / "predictions_cache_verified.pkl", "rb") as f:
        cache = pickle.load(f)
    t6 = pd.read_csv(DATA / "table06_ablation.csv").set_index("model")
    for model in ["mmap_full", "mmap_no_missingness"]:
        probs, targets = cache[model]
        recomputed = macro_auroc(probs, targets)
        reported = float(t6.loc[model, "macro_auroc"])
        match = close(recomputed, reported)
        ok &= match
        print(f"  {'OK' if match else 'CHECK'} {model:22s} recomputed={recomputed:.4f} reported={reported:.4f}")

    print("\n[2] Test prediction export vs Table 4 per-label AUROC")
    pred = pd.read_csv(DATA / "predictions_test_seed42.csv")
    t4 = pd.read_csv(DATA / "table04_canonical_performance.csv").set_index("label")
    for lab in LABELS:
        recomputed = roc_auc_score(pred[f"targ_{lab}"], pred[f"prob_{lab}"])
        reported = float(t4.loc[lab, "auroc"])
        match = close(recomputed, reported)
        ok &= match
        print(f"  {'OK' if match else 'CHECK'} {lab:16s} export={recomputed:.4f} reported={reported:.4f}")

    print("\n[3] Arithmetic checks for visible deltas")
    t9 = pd.read_csv(DATA / "table09_missingness_robustness.csv")
    for _, r in t9.iterrows():
        expected = round(float(r["auroc_30pct_mcar"]) - float(r["auroc_full_meta"]), 4)
        reported = round(float(r["auroc_drop_mmap"]), 4)
        match = expected == reported
        ok &= match
        print(f"  {'OK' if match else 'CHECK'} Table 9 MMAP {r['label']:14s} expected={expected:+.4f} reported={reported:+.4f}")
    t13 = pd.read_csv(DATA / "table13_external_device_check.csv")
    for _, r in t13.iterrows():
        expected = float(r["brset_auroc"]) - float(r["mbrset_auroc"])
        reported = float(r["absolute_decline"])
        match = abs(expected - reported) <= 0.001
        ok &= match
        print(f"  {'OK' if match else 'CHECK'} Table 13 {r['label']:24s} expected≈{expected:.4f} reported={reported:.3f}")

    print("\n" + "=" * 72)
    print("Overall:", "PASS" if ok else "CHECK VALUES")
    print("=" * 72)


if __name__ == "__main__":
    main()
