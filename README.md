# Uncertainty-Aware RETFound

Safety-centered evaluation of uncertainty-aware last-layer adaptation on frozen RETFound retinal features for referable diabetic retinopathy screening under dataset shift.

This repository supports the manuscript:

**Uncertainty-Aware Last-Layer Adaptation of RETFound for Referable Diabetic Retinopathy Screening Under Dataset Shift**

Author: **Karim Mardhani**  
Affiliation: Master of Science in Artificial Intelligence, University of Colorado Boulder  
Repository: <https://github.com/kmardhani/uncertainty-aware-retfound>

## Overview

This project evaluates whether uncertainty-aware last-layer heads can improve safety-oriented operating points for referable diabetic retinopathy screening when applied to frozen RETFound retinal features.

The study uses RETFound as a frozen feature encoder and compares several cached-feature heads:

- softmax linear head
- post-hoc temperature scaling
- variational Bayesian last-layer heads
- diagonal Laplace last-layer approximation
- SNGP-style cached-feature head

The main focus is not state-of-the-art accuracy. The focus is whether uncertainty-aware heads improve clinically relevant operating points, especially false negatives, sensitivity, calibration, threshold behavior, and selective referral under dataset shift.

## Main finding

The results support a safety-centered conclusion:

Uncertainty-aware last-layer heads can improve internal safety-oriented operating points, but false-negative reduction is not unique to Bayesian modeling. Threshold tuning can also reduce false negatives, often at high false-positive cost. Uncertainty signals that appear useful internally may also weaken or fail under second-dataset validation.

Therefore, trustworthy retinal screening claims should be evaluated using explicit safety-coverage tradeoffs and second-dataset validation, not aggregate accuracy alone.

## Datasets

The experiments use two public diabetic retinopathy fundus-image datasets:

| Dataset | Total | Non-referable | Referable | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|---:|
| APTOS 2019 | 3,662 | 2,175 | 1,487 | 2,930 | 366 | 366 |
| DDR | 12,522 | 6,896 | 5,626 | 8,765 | 1,878 | 1,879 |

Binary mapping:

- grades 0 and 1: non-referable diabetic retinopathy
- grades 2, 3, and 4: referable diabetic retinopathy

Raw datasets are not included in this repository.

## Key results

### APTOS internal validation

On APTOS, uncertainty-aware heads improved safety-oriented operating points.

The strongest full-coverage SNGP sensitivity-selected model achieved:

- sensitivity: 0.9805
- specificity: 0.8585
- balanced accuracy: 0.9195
- false negatives: 3
- false positives: 30

At approximately 80% accepted coverage, entropy-based selective referral for the APTOS SNGP sensitivity-selected checkpoint reduced accepted-case false negatives to zero while preserving high accepted-case specificity.

### DDR second-dataset validation

On DDR, native Bayesian heads qualitatively reproduced the APTOS direction, but with weaker tradeoffs.

The DDR Bayesian sensitivity-selected model reduced false negatives relative to the DDR softmax baseline:

- DDR softmax false negatives: 188
- DDR Bayesian sensitivity-selected false negatives: 129

However, specificity dropped from 0.8075 to 0.7031, showing the safety-specificity tradeoff.

### APTOS-to-DDR SNGP transfer

The APTOS-trained SNGP checkpoint transferred poorly to DDR without retraining:

- sensitivity: 0.2701
- AUC: 0.5833
- balanced accuracy: 0.5688
- false negatives: 616

This is an important negative result. It shows that internal uncertainty behavior does not necessarily imply robustness under dataset shift.

## Repository structure

```text
.
├── configs/                 # Dataset and experiment configuration files
├── paper/                   # Manuscript source, tables, figures, and build script
│   ├── figures/
│   ├── scripts/
│   ├── tables/
│   ├── main.tex
│   ├── references.bib
│   └── build.sh
├── scripts/                 # Data preparation, feature extraction, training, and analysis scripts
│   ├── analysis/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   └── training/
├── src/                     # Python package source
│   └── uncertainty_retfound/
├── tests/                   # Unit and script tests
├── pyproject.toml
├── uv.lock
└── README.md
```

Private working notes, intermediate decision logs, internal experiment logs, raw datasets, checkpoints, and large experiment outputs are intentionally not included in the public repository.

## Installation

This project uses `uv`.

```bash
git clone https://github.com/kmardhani/uncertainty-aware-retfound.git
cd uncertainty-aware-retfound
uv sync
```

Run tests:

```bash
uv run pytest
```

## Reproducing paper assets

The manuscript tables and figures can be regenerated from locked summary outputs with:

```bash
uv run python paper/scripts/make_paper_assets.py
```

Then build the paper:

```bash
cd paper
./build.sh
```

The generated PDF is ignored by Git.

## Experiment workflow

The project is organized around the following pipeline:

1. Prepare APTOS 2019 and DDR metadata splits.
2. Extract frozen RETFound features.
3. Train cached-feature heads.
4. Apply calibration and uncertainty estimation.
5. Run threshold sweeps.
6. Run selective-referral evaluation.
7. Build summary tables.
8. Generate manuscript assets.

Representative scripts:

```bash
uv run python -m scripts.data.prepare_aptos_metadata
uv run python -m scripts.data.prepare_ddr_metadata
uv run python -m scripts.features.export_retfound_features
uv run python -m scripts.training.train_feature_head
uv run python -m scripts.training.train_bayesian_feature_head
uv run python -m scripts.training.train_sngp_feature_head
uv run python -m scripts.analysis.evaluate_threshold_sweep
uv run python -m scripts.analysis.evaluate_selective_referral
uv run python -m scripts.analysis.build_experiment_summary_tables
```

Some full experiment commands require local paths to raw datasets, RETFound checkpoints, and generated feature files. Those artifacts are not committed to the repository.

## Public repository scope

This public repository contains:

- source code
- tests
- manuscript source
- paper tables and figures
- paper asset-generation scripts
- summary artifacts needed to reconstruct the reported manuscript tables and figures

This public repository does not include:

- raw APTOS or DDR images
- RETFound checkpoint files
- cached feature matrices
- trained model checkpoints
- full raw experiment output directories
- private research logs or working notes
- cluster-specific paths or logs

## Limitations

This is not a clinical deployment study.

The study is limited by:

- use of frozen RETFound features rather than raw-image fine-tuning
- limited uncertainty-method families
- evaluation on one second dataset rather than multiple independent external cohorts
- reliance on cached-feature experiments rather than full end-to-end retinal foundation model adaptation

The results should be interpreted as a research-grade empirical evaluation of uncertainty-aware last-layer adaptation, not as evidence of clinical readiness.

## Independent project status

This work was conducted as an independent research initiative by the author.

The author is enrolled in the part-time, course-based Master of Science in Artificial Intelligence program at the University of Colorado Boulder; however, this project was not sponsored, supervised, or formally endorsed by the university.

## Citation

A formal citation will be added after the arXiv version is available.

## License

A license has not yet been finalized. Until a license is added, reuse rights are not explicitly granted beyond viewing the public repository.
