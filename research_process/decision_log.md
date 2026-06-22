# Decision Log

## Decision 001 — Start with Frozen RETFound Embeddings

**Date:** 2026-06-22

### Decision

The initial experimental strategy will use RETFound as a frozen feature extractor. A deterministic last-layer classifier and one or more Bayesian last-layer variants will be trained on top of the extracted RETFound embeddings.

### Rationale

Starting with frozen RETFound embeddings keeps the first version of the project focused, reproducible, and easier to debug. It separates the central research question — whether Bayesian last-layer adaptation improves calibration, uncertainty estimation, selective referral, and robustness — from the additional complexity of full foundation-model fine-tuning.

This approach also allows faster iteration. Once RETFound embeddings are extracted, deterministic and Bayesian last-layer experiments can be repeated without repeatedly running the full image backbone.

### Consequences

- The first experiments will focus on last-layer adaptation rather than full model fine-tuning.
- Fine-tuning RETFound remains an optional extension.
- Saved embeddings will become an important intermediate artifact.
- Dataset preprocessing and feature extraction must be carefully documented.

### Status

Accepted.

## Decision 002 — Use a Staged Dataset Strategy

**Date:** 2026-06-22

### Decision

The project will use a staged dataset strategy rather than starting immediately with the largest available retinal dataset.

The initial pipeline will be validated using **APTOS 2019** or a small **EyePACS subset**. After the full pipeline is working, the project will scale to **full EyePACS** or a large EyePACS version using the available A100/H100 GPU cluster. External validation and dataset-shift testing will later use a compatible dataset such as **Messidor**, **Messidor-2**, **IDRiD**, or another diabetic retinopathy dataset.

### Rationale

This staged approach balances practical feasibility with research credibility.

Starting with a smaller dataset or subset allows faster iteration on the core research pipeline:

- data loading
- preprocessing
- RETFound feature extraction
- deterministic baseline training
- Bayesian last-layer adaptation
- calibration evaluation
- uncertainty scoring
- selective referral analysis

Scaling later to EyePACS strengthens the academic quality of the work by providing a larger and more diverse dataset for the main experiments. However, dataset size alone does not make the project research-grade. The main contribution depends on rigorous evaluation of calibration, uncertainty estimation, selective referral, and dataset-shift robustness.

### Consequences

- The first implementation will prioritize pipeline correctness over dataset scale.
- APTOS 2019 or a small EyePACS subset will be used for early development.
- Full EyePACS will be treated as the likely main dataset after the pipeline is stable.
- A100/H100 cluster access will be used for full-dataset RETFound embedding extraction and larger experiments.
- The MacBook Pro M3 Pro with 18 GB memory will be used for local development, smoke tests, documentation, and analysis.
- External datasets will be considered later for dataset-shift evaluation.

### Status

Accepted.