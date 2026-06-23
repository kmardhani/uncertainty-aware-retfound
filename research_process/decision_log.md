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

## Decision 003 — Use RETFound as a Frozen Feature Extractor First

**Date:** 2026-06-22

### Decision

The first implementation will use RETFound as a frozen feature extractor. The RETFound backbone will be loaded with pretrained weights, kept frozen, and used to extract image embeddings. Deterministic and Bayesian last-layer classifiers will then be trained on top of the saved embeddings.

### Rationale

The central research question is whether Bayesian last-layer adaptation improves calibration, uncertainty estimation, selective referral, and robustness under dataset shift.

Starting with frozen RETFound embeddings keeps the initial experiments focused on this question. It avoids mixing the effects of Bayesian last-layer adaptation with the additional complexity of full or partial foundation-model fine-tuning.

This strategy also supports faster experimentation. Once embeddings are extracted, multiple last-layer methods can be trained and evaluated without repeatedly running the full RETFound image backbone.

### Consequences

- The first experiments will focus on last-layer adaptation rather than end-to-end fine-tuning.
- RETFound embedding extraction becomes a key intermediate step.
- Saved embeddings must include metadata such as dataset, split, label mapping, checkpoint identifier, preprocessing version, and git commit hash.
- Full or partial RETFound fine-tuning remains a later extension.
- The A100/H100 cluster will be used for full-dataset embedding extraction.
- The local MacBook Pro M3 Pro will be used mainly for development, smoke tests, and analysis.

### Status

Accepted.

## Decision 004 - Use a Tested Metadata Preparation Script Before Model Integration

**Date:** 2026-06-22  

**Status:** Accepted

### Context

The project is moving from repository scaffolding into dataset preparation for APTOS 2019 diabetic retinopathy classification. Before integrating RETFound or uncertainty-aware model components, the project needs a reliable and reproducible way to prepare metadata for experiments.

The initial dataset preparation workflow needs to support:

- Loading a dataset YAML configuration.

- Loading APTOS metadata from CSV.

- Applying task-specific label mappings.

- Generating train/validation/test splits.

- Saving split metadata for later training and evaluation.

- Testing the workflow with fixture data.

### Decision

Create a dedicated script:

```text

scripts/data/prepare_aptos_metadata.py
```

## Decision 005 - Validate APTOS Image Paths Before Dataset Class and Model Training

**Date:** 2026-06-22  

**Status:** Accepted

### Context

After implementing metadata preparation for APTOS 2019, the next risk was whether metadata rows could reliably resolve to local retinal image files.

Before adding a PyTorch Dataset class or integrating RETFound, the project needed a lightweight validation step that confirms image file paths exist without loading all images into memory.

### Decision

Add image path resolution and validation functionality before implementing model-facing dataset loading.

The validation workflow checks whether each metadata row resolves to an expected image path and reports:

- Total metadata rows.

- Number of image files found.

- Number of image files missing.

- Missing image identifiers or paths, capped for readability.

A CLI script was added:

```text

scripts/data/validate_aptos_images.py
```

## Decision: Add a Lightweight APTOS Dataset Wrapper Before RETFound Integration

**Date:** 2026-06-22  

**Status:** Accepted

### Context

After implementing metadata preparation and image path validation, the next step was to create a model-facing interface for reading APTOS examples.

The project needs a dataset abstraction that can load individual images and labels from prepared metadata. However, full RETFound integration and PyTorch training are not yet ready.

### Decision

Add an `APTOSDataset` wrapper that supports dataframe or CSV metadata input, resolves image paths, opens images with Pillow, converts images to RGB, applies an optional transform, and returns structured samples.

The dataset currently returns dictionaries containing:

```text

image

label

image_path

id_code