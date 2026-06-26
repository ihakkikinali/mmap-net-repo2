# Trained weights

Trained checkpoints (seeds 42, 123, 777) are **not** redistributed in this
review package because of file size. They will be archived in a public
repository upon acceptance.

The per-sample prediction caches under `../data/` are sufficient to verify the
reported Protocol-2 ablation table (see `../scripts/verify_consistency.py`).
The full model loads with `strict=True` at 69,369,413 parameters.
