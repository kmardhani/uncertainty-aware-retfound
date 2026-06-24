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

## Decision 008 — Treat Sensitivity-Selected Bayesian Heads As Screening-Oriented Operating Points

**Date:** 2026-06-24

### Decision

Sensitivity-selected variational Bayesian cached-feature heads should be interpreted as screening-oriented operating points, not as blanket replacements for the temperature-scaled deterministic baseline.

### Context

The completed sensitivity-selected run is:

```text
outputs/feature_heads/retfound_variational_bayesian_20epoch_best_sensitivity
```

using the same Bayesian setup as the earlier run:

1. Variational Bayesian linear head on cached RETFound features
2. `20` epochs
3. Batch size `8`
4. Learning rate `0.001`
5. `mc_samples_train=1`
6. `mc_samples_eval=30`
7. `prior_std=1.0`
8. `kl_weight=1/2930`

with:

1. `selection_metric=sensitivity`
2. Best epoch `7`

Best validation metrics:

1. Accuracy: `0.8962`
2. AUC: `0.9617`
3. Sensitivity: `0.9545`
4. Specificity: `0.8538`
5. Balanced accuracy: `0.9042`
6. ECE: `0.0305`
7. NLL: `0.2710`
8. Brier: `0.0816`
9. Confusion matrix: `[[181, 31], [7, 147]]`

Relative to the cached softmax baseline with temperature scaling:

1. Sensitivity improved from `0.8896` to `0.9545`
2. False negatives decreased from `17` to `7`
3. Accuracy improved
4. AUC improved
5. ECE, NLL, and Brier became worse

The run also continued to separate correct and incorrect predictions in uncertainty space:

1. Mean confidence: correct `0.8883` vs incorrect `0.7248`
2. Mean predictive entropy: correct `0.2789` vs incorrect `0.5376`
3. Mean probability variance: correct `0.0021` vs incorrect `0.0053`
4. Mean mutual information: correct `0.0067` vs incorrect `0.0130`

However, the persistent hard false-negative case `025a169a0bb0` remained, with confidence reduced to `0.8739` versus `0.9291` in the val-loss-selected Bayesian model and `0.9720` in the final epoch model.

### Rationale

This run is valuable because it shows that the Bayesian head can move to a higher-sensitivity operating point while keeping useful uncertainty separation. That is relevant for screening use cases, where missed positives are especially costly.

At the same time, weaker ECE, NLL, and Brier scores mean the model is not simply better overall. The right interpretation is therefore operating-point-dependent:

1. It is a stronger candidate when sensitivity is the primary target
2. It is not automatically the preferred choice when calibration quality is the main target
3. It still requires selective-referral and high-confidence-error analysis

### Consequences

- Future reporting should distinguish between val-loss-selected and sensitivity-selected Bayesian operating points.
- Screening-oriented comparisons should explicitly report false-negative counts.
- The persistent hard false-negative case remains a required qualitative analysis item.
- The next useful step is either comparing multiple epoch-selection metrics systematically or implementing the Laplace last-layer baseline.

### Status

Accepted.

## Decision 009 — Use Bayesian Hyperparameters To Explore Operating-Point Tradeoffs, Not To Search For A Universal Winner

**Date:** 2026-06-24

### Decision

Bayesian hyperparameter sweeps should be interpreted as a way to map operating-point tradeoffs among sensitivity, specificity, calibration, and uncertainty behavior, rather than as a search for one universally best model.

### Context

The Bayesian sweep is stored at:

```text
outputs/feature_heads/sweeps/
```

It varied:

1. `kl_weight` in `[0.00003, 0.0001, 0.000341, 0.001]`
2. `prior_std` in `[0.5, 1.0, 2.0]`

All runs used:

1. `selection_metric=sensitivity`
2. `20` epochs
3. Batch size `8`
4. Learning rate `0.001`
5. `mc_samples_train=1`
6. `mc_samples_eval=30`

Two informative sweep candidates were:

Maximum-sensitivity candidate:

```text
outputs/feature_heads/sweeps/bayes_sensitivity_kl_0.001_prior_2.0
```

1. Best epoch `7`
2. Accuracy `0.8934`
3. AUC `0.9613`
4. Sensitivity `0.9610`
5. Specificity `0.8443`
6. Balanced accuracy `0.9027`
7. ECE `0.0320`
8. NLL `0.2794`
9. Brier `0.0842`
10. Confusion matrix `[[179, 33], [6, 148]]`

Balanced screening candidate:

```text
outputs/feature_heads/sweeps/bayes_sensitivity_kl_0.00003_prior_2.0
```

1. Best epoch `7`
2. Accuracy `0.9016`
3. AUC `0.9617`
4. Sensitivity `0.9545`
5. Specificity `0.8632`
6. Balanced accuracy `0.9089`
7. ECE `0.0275`
8. NLL `0.2639`
9. Brier `0.0796`
10. Confusion matrix `[[183, 29], [7, 147]]`

Relative to the cached softmax plus temperature-scaled baseline with sensitivity `0.8896` and `17` false negatives:

1. The maximum-sensitivity Bayesian candidate reduced false negatives to `6`
2. The balanced Bayesian candidate reduced false negatives to `7`
3. The balanced Bayesian candidate also improved accuracy and balanced accuracy

The maximum-sensitivity candidate also retained useful uncertainty separation:

1. Mean confidence: correct `0.8853` vs incorrect `0.7253`
2. Mean predictive entropy: correct `0.2841` vs incorrect `0.5356`
3. Mean probability variance: correct `0.0055` vs incorrect `0.0139`
4. Mean mutual information: correct `0.0179` vs incorrect `0.0344`

The persistent hard false-negative case `025a169a0bb0` still remained incorrect, although its confidence decreased to `0.8498`.

### Rationale

The sweep shows that Bayesian hyperparameters materially affect the screening-oriented operating point. Higher-regularization and larger-prior settings can push the model toward fewer false negatives, but that comes with real tradeoffs in specificity and calibration summaries. A more balanced setting may recover some overall performance while giving up some of the strongest uncertainty separation.

This means model choice should stay tied to the intended operating goal:

1. Maximum-sensitivity setting when missed positives are the main concern
2. More balanced setting when overall operating-point quality matters more
3. Neither setting should be described as universally best

### Consequences

- Future sweep reports should describe tradeoffs explicitly instead of only reporting the single top metric.
- False-negative counts should remain a first-class comparison target.
- Uncertainty separation should be tracked alongside calibration summaries, because the best classification tradeoff may not yield the strongest uncertainty signal.
- The next step remains either deeper selection-metric comparison or implementation of the Laplace last-layer baseline.

### Status

Accepted.

## Decision 010 — Treat Diagonal Laplace As A Bayesian Baseline, Not The Leading Calibration Method

**Date:** 2026-06-24

### Decision

The diagonal Laplace last-layer approximation will be retained as a useful Bayesian baseline, but current evidence does not support treating it as the preferred calibration method over variational Bayesian or temperature scaling.

### Context

The Laplace implementation uses:

1. A deterministic cached-feature linear head trained first
2. A diagonal Laplace posterior approximation over the final layer afterward

Initial val-loss-selected Laplace run:

```text
outputs/feature_heads/retfound_laplace_20epoch_best_val_loss
```

1. Best epoch `15`
2. Accuracy `0.8962`
3. AUC `0.9666`
4. Sensitivity `0.8636`
5. Specificity `0.9198`
6. Balanced accuracy `0.8917`
7. ECE `0.0972`
8. NLL `0.2891`
9. Brier `0.0826`
10. Confusion matrix `[[195, 17], [21, 133]]`

Initial sensitivity-selected Laplace run:

```text
outputs/feature_heads/retfound_laplace_20epoch_best_sensitivity
```

1. Best epoch `6`
2. Accuracy `0.8962`
3. AUC `0.9574`
4. Sensitivity `0.9416`
5. Specificity `0.8632`
6. Balanced accuracy `0.9024`
7. ECE `0.1231`
8. NLL `0.3349`
9. Brier `0.0971`
10. Confusion matrix `[[183, 29], [9, 145]]`

Laplace prior-precision sweep:

```text
outputs/feature_heads/sweeps/laplace_sensitivity_prior_precision_*/
```

with `prior_precision` in `[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]`.

Best sensitivity Laplace result:

1. `prior_precision=0.03`
2. Best epoch `6`
3. Sensitivity `0.9481`
4. Specificity `0.8632`
5. Accuracy `0.8989`
6. AUC `0.9585`
7. Balanced accuracy `0.9056`
8. ECE `0.1382`
9. NLL `0.3454`
10. Brier `0.1002`
11. Confusion matrix `[[183, 29], [8, 146]]`

Better-calibrated Laplace sweep candidate:

1. `prior_precision=10.0`
2. Best epoch `6`
3. Sensitivity `0.9481`
4. Specificity `0.8585`
5. Accuracy `0.8962`
6. AUC `0.9586`
7. Balanced accuracy `0.9033`
8. ECE `0.0900`
9. NLL `0.3061`
10. Brier `0.0894`
11. Confusion matrix `[[182, 30], [8, 146]]`

Relative to the cached softmax plus temperature-scaled baseline, the best sensitivity Laplace setting improved false negatives from `17` to `8`, but had worse ECE, NLL, and Brier score.

Relative to the variational Bayesian results, variational Bayesian remained stronger on the sensitivity-balanced-screening tradeoff, with `6` or `7` false negatives depending on operating point.

### Rationale

This is enough evidence to keep Laplace in the comparison set, because it is a legitimate Bayesian uncertainty baseline and it does improve screening-style error counts over the deterministic baseline.

However, the diagonal approximation did not outperform variational Bayesian or temperature scaling on calibration-oriented metrics, and it was not the strongest method on the full sensitivity-specificity tradeoff either.

### Consequences

- Laplace should remain in the reported baseline set.
- Variational Bayesian remains the stronger current Bayesian method for cached-feature screening experiments.
- Temperature scaling remains the stronger simple calibration baseline.
- Future work on Laplace would need either a better curvature approximation or stronger posterior modeling to change this conclusion.

### Status

Accepted.

## Decision 011 — Prefer The Best Single Bayesian Operating Point Over Simple Ensemble Referral Rules

**Date:** 2026-06-24

### Decision

The project will treat the strongest single Bayesian operating points as the primary screening baselines rather than simple ensemble referral rules such as OR, majority vote, or AND.

### Context

Decision-policy outputs were written to:

1. `outputs/decision_policies/retfound_policy_comparison.json`
2. `outputs/decision_policies/retfound_policy_comparison.csv`

Compared models and policies:

1. `softmax_temp`
2. `bayes_max_sensitivity`
3. `bayes_balanced`
4. `laplace_sensitivity`
5. `OR rule`
6. `majority vote`
7. `AND rule`

Key single-model results:

1. `softmax_temp`: sensitivity `0.8896`, specificity `0.8726`, false negatives `17`, false positives `27`, referral_rate `0.4481`
2. `bayes_max_sensitivity`: sensitivity `0.9610`, specificity `0.8443`, false negatives `6`, false positives `33`, referral_rate `0.4945`
3. `bayes_balanced`: sensitivity `0.9545`, specificity `0.8632`, false negatives `7`, false positives `29`, referral_rate `0.4809`
4. `laplace_sensitivity`: sensitivity `0.9481`, specificity `0.8585`, false negatives `8`, false positives `30`, referral_rate `0.4809`

Key ensemble and rule-based results:

1. `OR rule`: sensitivity `0.9610`, specificity `0.8302`, false negatives `6`, false positives `36`, referral_rate `0.5027`
2. `majority vote`: sensitivity `0.9545`, specificity `0.8491`, false negatives `7`, false positives `32`, referral_rate `0.4891`
3. `AND rule`: sensitivity `0.8896`, specificity `0.8915`, false negatives `17`, false positives `23`, referral_rate `0.4372`

The `OR rule` did not reduce false negatives below `bayes_max_sensitivity`; both had `6`. It also increased false positives and referral rate. `majority vote` did not improve over `bayes_balanced`; it matched false negatives but produced more false positives. The `AND rule` improved specificity but was not suitable for screening because it lost sensitivity.

### Rationale

This comparison suggests that the remaining misses are at least partly shared hard cases across models. If that is true, then naive aggregation rules will not improve safety unless they add a genuinely different error profile.

The strongest current operating points therefore remain the single Bayesian models that were already tuned for the target use case:

1. `bayes_max_sensitivity` for maximum-sensitivity screening
2. `bayes_balanced` for a more balanced screening tradeoff

### Consequences

- Future screening comparisons should keep `bayes_max_sensitivity` as the primary maximum-sensitivity baseline.
- Future balanced comparisons should keep `bayes_balanced` as the primary balanced-screening baseline.
- Simple OR/majority/AND ensemble policies should not be treated as a safety improvement by default.
- Any future ensemble work should focus on genuinely complementary models or richer decision logic, not only naive voting rules.

### Status

Accepted.

## Decision 012 — Treat Selective Referral As The Strongest Current Safety Lever

**Date:** 2026-06-24

### Decision

Selective referral based on uncertainty ranking should be treated as the strongest current safety-oriented result in the project, and it should be prioritized over naive ensemble voting for follow-up analysis.

### Context

Selective referral was evaluated with:

```text
scripts/analysis/evaluate_selective_referral.py
```

using:

```text
outputs/feature_heads/sweeps/bayes_sensitivity_kl_0.001_prior_2.0/best_validation_predictions.csv
```

Generated outputs:

1. `outputs/selective_referral/bayes_max_sensitivity_confidence.csv`
2. `outputs/selective_referral/bayes_max_sensitivity_predictive_entropy.csv`
3. `outputs/selective_referral/bayes_max_sensitivity_probability_variance.csv`
4. `outputs/selective_referral/bayes_max_sensitivity_mutual_information.csv`

Baseline without deferral:

1. Coverage `1.0`
2. Accuracy `0.8934`
3. Sensitivity `0.9610`
4. Specificity `0.8443`
5. False negatives `6`
6. False positives `33`

Selected referral results:

Confidence / predictive entropy:

1. Around `90%` coverage: deferred `36`, accepted `330`, false negatives `3`, sensitivity `0.9790`, specificity `0.8663`, accuracy `0.9152`
2. Around `80%` coverage: deferred `73`, accepted `293`, false negatives `1`, sensitivity `0.9922`, specificity `0.9085`, accuracy `0.9454`
3. Around `70%` coverage: deferred `109`, accepted `257`, false negatives `1`, sensitivity `0.9914`, specificity `0.9291`, accuracy `0.9572`

Mutual information:

1. Around `80%` coverage: deferred `73`, accepted `293`, false negatives `1`, sensitivity `0.9924`, specificity `0.8696`, accuracy `0.9249`
2. Around `70%` coverage: deferred `109`, accepted `257`, false negatives `0`, sensitivity `1.0000`, specificity `0.8944`, accuracy `0.9416`

Probability variance:

1. Around `80%` coverage: deferred `73`, accepted `293`, false negatives `1`, sensitivity `0.9925`, specificity `0.9057`, accuracy `0.9454`

### Rationale

This is the clearest current evidence that uncertainty is useful for safety-oriented decision support in this project.

The main result is not that the classifier became universally better. The main result is that deferring uncertain cases can materially reduce false negatives among the accepted automated decisions. That is more aligned with a real screening workflow than simply comparing full-coverage summary metrics.

At the same time, this only shifts part of the burden to human review. Accepted-case metrics are not population-wide performance metrics, and the deferred cases still need clinical handling.

### Consequences

- Selective-referral analysis should become a primary evaluation axis for future Bayesian and Laplace comparisons.
- Naive ensemble voting is lower priority than uncertainty-based defer-to-human policies.
- Future reporting must clearly separate accepted-case performance from full-population performance.
- These results should be described as offline validation evidence for triage-style workflows, not as clinical validation.

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


## Decision: Treat Threshold Tuning as a Required Screening Control

**Date:** 2026-06-24

**Status:** Accepted

### Context

The project had already shown that Bayesian last-layer models and uncertainty-aware selective referral could reduce false negatives on APTOS validation data. However, that alone was not enough to support a strong causal claim, because an ordinary probability-threshold change might achieve similar false-negative reduction without Bayesian uncertainty.

Threshold-sweep analysis was added through `scripts/analysis/evaluate_threshold_sweep.py` and applied to:

- `outputs/threshold_sweeps/softmax_temp_threshold_sweep.csv`
- `outputs/threshold_sweeps/bayes_max_sensitivity_threshold_sweep.csv`
- `outputs/threshold_sweeps/bayes_balanced_threshold_sweep.csv`

At their zero-false-negative operating points, the temperature-scaled softmax model and both Bayesian models all converged to the same validation behavior: sensitivity `1.0000`, specificity `0.5849`, `88` false positives, referral rate `0.6612`, and accuracy `0.7596`.

### Decision

Treat threshold tuning as a required screening control and narrow the study claim accordingly.

The project should not claim that Bayesian last-layer methods are required to achieve zero false negatives on the current validation set. Instead, the defensible claim is that Bayesian uncertainty is useful for selective-referral workflows that preserve substantially stronger accepted-case accuracy and specificity while routing uncertain cases to human review.

### Rationale

This change improves causal discipline in the study narrative.

If a low decision threshold can remove false negatives for both deterministic and Bayesian models, then zero false negatives alone is not evidence that Bayesian uncertainty is the unique mechanism. The more meaningful comparison is between two different workflows:

- low-threshold automatic screening, which expands positive referrals directly
- uncertainty-aware selective referral, which defers uncertain cases for human review

The selective-referral result remains stronger on clinical-safety grounds because it achieves better accepted-case accuracy and specificity with an explicit review pathway, rather than simply broadening the automated positive region.

### Consequences

Positive consequences:

- The project now has a stronger control against overclaiming Bayesian necessity.

- The interpretation of uncertainty methods is more precise.

- Future comparisons can distinguish thresholding effects from true uncertainty-routing benefits.

Limitations:

- The threshold-sweep conclusion is still based on a single validation set.

- Threshold tuning and selective referral are not interchangeable workflows, so they should not be compared on false negatives alone.

- External validation is still required before drawing broader screening conclusions.

### Follow-up Actions

- Use threshold-sweep tables as a standard control in future screening analyses.

- Report human-review burden alongside sensitivity-focused operating points.

- Continue treating selective referral, not zero false negatives alone, as the main safety-oriented Bayesian result.
