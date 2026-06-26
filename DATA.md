# Data and released artifacts

This repository provides reproducibility artifacts that do not require redistribution of raw retinal images.

## Included

- Aggregate manuscript tables in `data/table*.csv`.
- Held-out BRSET test prediction export for the four primary labels.
- Validation-split probability export used to fit validation-fitted isotonic calibration and score-calibrated deferral thresholds.
- Hashed patient-level split map.
- Quantitative figures in both PNG and PDF format.

## Not included

- Raw BRSET or mBRSET images.
- Original image filenames or patient identifiers.
- Full trained checkpoints.
- Full raw training notebooks from the development environment.

## Scope boundaries

The missingness experiments are controlled stress tests and sanity checks. They should not be interpreted as validation of a real-world audited MNAR mechanism. The mBRSET analysis is a single external-device shift check, not evidence of broad multi-country or multi-camera generalization.
