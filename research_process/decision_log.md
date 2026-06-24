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
```

## Decision: Add Pillow-Based Image Transforms Before PyTorch Integration

**Date:** 2026-06-22  

**Status:** Accepted

### Context

The project now has metadata preparation, image path validation, and a lightweight APTOS dataset wrapper. The next requirement is image preprocessing.

However, the project is not yet ready for full PyTorch, torchvision, RETFound, or training-loop integration.

### Decision

Add a small Pillow-based preprocessing layer that works with the existing `APTOSDataset` transform interface.

The preprocessing layer includes:

```text

EnsureRGB

ResizeImage

CenterCropImage

ComposeTransforms

build_transform_from_config
```

## Decision: Add PyTorch DataLoader Utilities Before Model Training

**Date:** 2026-06-22  

**Status:** Accepted

### Context

The project had a tested data preparation pipeline, image path validation, dataset wrapper, and Pillow-based preprocessing transforms. The next step was to convert dataset samples into tensor batches that future training code can consume.

### Decision

Add PyTorch and torchvision dependencies, update `APTOSDataset` to inherit from `torch.utils.data.Dataset`, add torchvision transform construction, and add a minimal dataloader utility.

The dataloader utility is intentionally small and does not include training logic.

### Rationale

This creates the necessary bridge between the data pipeline and future model training while keeping the scope limited.

Adding PyTorch is now justified because the project has reached the point where batch tensor creation is needed. PyTorch was intentionally not added earlier to avoid premature dependency expansion.

### Consequences

Positive consequences:

- Dataset samples can now be batched for training.

- Image tensors can be produced through torchvision transforms.

- Batch outputs preserve image paths and image identifiers.

- The project is ready for a baseline model/training skeleton.

- The full test suite now passes with 33 tests.

Limitations:

- No model is implemented yet.

- No training loop is implemented yet.

## Decision 006 — Use Cached-Feature Deterministic Baselines As The Primary Reference For Bayesian Heads

**Date:** 2026-06-24

### Decision

Future Bayesian-head and Laplace-head comparisons will use the cached-feature softmax linear-head baseline and the cached-feature temperature-scaled baseline as the primary deterministic reference points.

### Context

The frozen RETFound feature export milestone is now complete for APTOS 2019. Exported cached features are stored at:

```text
outputs/features/aptos2019_retfound_mae_natureCFP/
```

with shapes:

1. `train`: `(2930, 1024)`
2. `val`: `(366, 1024)`
3. `test`: `(366, 1024)`

The first completed cached-feature softmax linear-head run is:

```text
outputs/feature_heads/retfound_softmax_linear_5epoch_bs8
```

with validation results:

1. Accuracy: `0.8798`
2. AUC: `0.9580`
3. Sensitivity: `0.8896`
4. Specificity: `0.8726`
5. NLL: `0.2643`
6. Brier: `0.0807`
7. ECE: `0.0348`
8. Confusion matrix: `[[185, 27], [17, 137]]`

Temperature scaling then produced:

1. Learned temperature: `0.7649`
2. ECE improvement from `0.0348` to `0.0186`
3. NLL improvement from `0.2643` to `0.2558`
4. Brier improvement from `0.0807` to `0.0798`
5. No change in classification metrics

### Rationale

The Bayesian-head and Laplace-head methods planned for this project operate on frozen RETFound features. The fairest deterministic baseline is therefore the matching cached-feature softmax head, not only the image-based baseline.

This does not mean the cached-feature and image-based baselines are identical. They are comparable, but not strictly the same experiment. Differences are expected because the cached-feature path fixes the encoder outputs in advance and changes the optimization dynamics.

The batch-size comparison also showed that batch size 32 undertrained relative to batch size 8 because it produced fewer optimizer updates over the same epoch budget. This reinforces the need to compare future methods against a well-tuned cached-feature deterministic baseline rather than a weaker large-batch reference.

### Consequences

- Future deterministic-vs-Bayesian comparisons should report both the raw cached-feature softmax baseline and the temperature-scaled cached-feature baseline.
- Image-based baselines remain useful context, but they are secondary comparison points for frozen-feature methods.
- Batch-size comparisons for cached-feature heads should account for optimizer-update count, not only epoch count.

## Decision 007 — Evaluate Bayesian Cached-Feature Heads Against The Temperature-Scaled Deterministic Baseline

**Date:** 2026-06-24

### Decision

The first variational Bayesian cached-feature head result will be interpreted primarily against the cached softmax plus temperature-scaled baseline, with explicit attention to both aggregate metrics and high-confidence failure modes.

### Context

The first completed Bayesian run is:

```text
outputs/feature_heads/retfound_variational_bayesian_20epoch_best_val_loss
```

using:

1. A variational Bayesian linear head on cached RETFound features
2. `20` epochs
3. Batch size `8`
4. Learning rate `0.001`
5. `mc_samples_train=1`
6. `mc_samples_eval=30`
7. `prior_std=1.0`
8. `kl_weight=1/2930`
9. `selection_metric=val_loss`

Best epoch:

1. Epoch `16`

Best validation metrics:

1. Accuracy: `0.8934`
2. AUC: `0.9650`
3. Sensitivity: `0.8831`
4. Specificity: `0.9009`
5. Balanced accuracy: `0.8920`
6. ECE: `0.0216`
7. NLL: `0.2371`
8. Brier: `0.0724`
9. Confusion matrix: `[[191, 21], [18, 136]]`

Relative to the cached softmax baseline with temperature scaling, the Bayesian run improved accuracy, AUC, NLL, Brier, and specificity, while showing slightly worse ECE and slightly lower sensitivity.

The uncertainty summaries also separated correct from incorrect predictions:

1. Mean confidence: correct `0.9084` vs incorrect `0.6929`
2. Mean predictive entropy: correct `0.2375` vs incorrect `0.5682`
3. Mean probability variance: correct `0.0064` vs incorrect `0.0235`
4. Mean mutual information: correct `0.0224` vs incorrect `0.0572`

At the same time, one high-confidence false negative remained.

### Rationale

This result is strong enough to justify continuing with Bayesian last-layer comparisons, but not strong enough to justify a blanket claim that the Bayesian head is safer or uniformly better than the deterministic baseline.

The comparison standard therefore needs to remain explicit:

1. Compare against the temperature-scaled deterministic cached-feature baseline
2. Track both calibration and discrimination metrics
3. Examine high-confidence errors directly
4. Prioritize selective-referral analysis rather than relying only on global averages

### Consequences

- Future Bayesian comparisons should include explicit reference to the temperature-scaled deterministic cached-feature baseline.
- High-confidence false negatives remain a required analysis target.
- The next implementation step can reasonably be either alternative epoch-selection criteria or a Laplace last-layer baseline.

### Status

Accepted.

### Status

Accepted.

- No metrics or calibration logic is implemented yet.

- RETFound integration is still pending.

### Follow-up Actions

- Add a minimal baseline model.

- Add a one-batch training-step test.

- Add simple classification loss handling.

- Add baseline evaluation utilities before RETFound-specific work.


## Decision: Establish Baseline Training Infrastructure Before RETFound Integration

**Date:** 2026-06-22  

**Status:** Accepted

### Context

The project had already established a tested data pipeline for APTOS metadata preparation, image path validation, image loading, preprocessing transforms, tensor conversion, and PyTorch DataLoader batching.

Before integrating RETFound, the project needed to prove that the training and evaluation stack works end-to-end using a simple baseline model.

### Decision

Add a minimal baseline training and evaluation infrastructure before RETFound integration.

This includes:

- A small CNN classifier.

- A one-batch training step.

- A reusable one-epoch training loop.

- A reusable evaluation loop.

- Basic classification metrics.

The baseline model is intentionally simple and is not intended to be the final research model.

### Rationale

This decision reduces integration risk.

RETFound integration will be easier and safer if the repository already has a working training path, evaluation path, and metric interface. If there are problems later, it will be easier to separate model-specific issues from general training infrastructure issues.

This also supports research-grade development by ensuring that each major layer of the project is testable before the next layer is added.

### Consequences

Positive consequences:

- The project now has a tested training smoke path.

- The project has a reusable epoch-level training loop.

- The project has a reusable evaluation loop.

- The project has basic classification metrics.

- The project can verify forward pass, loss, backward pass, optimizer step, evaluation outputs, and metrics using fake data.

- The full test suite now passes with 49 tests.

Limitations:

- The baseline model is intentionally small and not clinically meaningful.

- No real APTOS training has been run yet.

- No model checkpointing exists yet.

- No experiment tracking exists yet.

- No calibration or uncertainty metrics exist yet.

- RETFound integration is still pending.

### Follow-up Actions

- Download and validate the full APTOS 2019 dataset locally.

- Add a real baseline training CLI.

- Add checkpointing and experiment output structure.

- Add calibration metrics.

- Add uncertainty metrics.

- Integrate RETFound after the baseline pipeline is validated on real data.
