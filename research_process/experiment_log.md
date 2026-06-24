## Experiment: APTOS Metadata Preparation Smoke Test

**Date:** 2026-06-22  
**Status:** Completed  
**Related commits:**
- `0541a7e` Add APTOS metadata fixture
- `e26aae1` Add dataset split generation utility
- `2914292` Add APTOS metadata preparation script
- `a22da3e` Add dataset split utility and tests
- `1c6fa5b` Add tests for APTOS metadata preparation script

### Objective

Validate the first end-to-end dataset preparation step for the APTOS 2019 diabetic retinopathy dataset using a small fixture dataset.

The goal was to confirm that the project can:

1. Load an APTOS dataset YAML configuration.
2. Load APTOS metadata from CSV.
3. Apply a selected disease classification task mapping.
4. Generate train/validation/test splits.
5. Save prepared metadata as a CSV file.
6. Protect this workflow with automated tests.

### Setup

The smoke test used the fixture configuration:

```text
configs/datasets/fixtures/aptos2019_sample.yaml
```

## Experiment: APTOS Image Path Validation Smoke Test

**Date:** 2026-06-22  
**Status:** Completed  
**Related commit:**
- `84487de` Add APTOS image path validation

### Objective

Validate that APTOS metadata rows can be resolved to expected retinal image file paths without loading image pixels into memory.

This step was added before model training to ensure that prepared metadata can be connected to local image files in a reproducible and testable way.

### Scope

The validation workflow checks:

1. Image path construction from metadata rows.
2. Use of an image root directory.
3. Use of the configured image identifier column.
4. Use of the expected image file extension.
5. Detection of missing image files.
6. Reporting of found and missing image counts.
7. CLI behavior for successful and failed validation cases.

### Implementation

Image path resolution and validation functionality was added to:

```text
src/uncertainty_retfound/data/aptos.py
```

## Experiment: APTOS Dataset Wrapper Smoke Test

**Date:** 2026-06-22  

**Status:** Completed  

**Related commit:**

- `bd5ad1b` Add APTOS dataset wrapper

### Objective

Validate a lightweight dataset wrapper for loading APTOS image-label examples from prepared metadata.

This step creates the bridge between metadata preparation and future model training while avoiding premature RETFound or PyTorch training integration.

### Scope

The dataset wrapper supports:

1. Loading prepared metadata from a pandas dataframe.

2. Loading prepared metadata from a CSV file.

3. Resolving image paths from metadata rows.

4. Opening individual image files.

5. Converting images to RGB.

6. Returning labels from the configured label column.

7. Applying an optional transform.

8. Returning structured samples containing image, label, image path, and image identifier.

9. Early validation of missing image paths when enabled.

### Implementation

The dataset wrapper was added to:

```text

src/uncertainty_retfound/data/aptos.py
```

## Experiment: APTOS Image Preprocessing Transform Smoke Test

**Date:** 2026-06-22  

**Status:** Completed  

**Related commits:**

- `8d1d4ce` Add APTOS image preprocessing transforms

- `1174820` Fixed static validation errors

### Objective

Validate a small, deterministic image preprocessing layer for APTOS images before introducing PyTorch, torchvision, RETFound, or model training code.

This step ensures that image preprocessing behavior is explicit, testable, and compatible with the existing `APTOSDataset` transform interface.

### Scope

The preprocessing layer supports:

1. Ensuring images are in RGB mode.

2. Resizing images to a square or rectangular target size.

3. Center-cropping images to a square or rectangular target size.

4. Composing multiple transforms in order.

5. Building a simple transform pipeline from configuration.

6. Passing transforms into `APTOSDataset`.

### Implementation

The transform utilities were added to:

```text

src/uncertainty_retfound/data/transforms.py
```
## Experiment: PyTorch DataLoader Smoke Test

**Date:** 2026-06-22  

**Status:** Completed  

**Related commit:**

- `8578751` Add PyTorch dataloader utilities

### Objective

Validate that APTOS image-label examples can be converted into PyTorch tensor batches suitable for future model training.

This step bridges the existing PIL-based dataset layer into the PyTorch training ecosystem while still avoiding model, training-loop, and RETFound integration.

### Scope

The implementation supports:

1. `APTOSDataset` inheritance from `torch.utils.data.Dataset`.

2. Torchvision-based transform construction.

3. Tensor conversion for images.

4. Optional image normalization.

5. PyTorch `DataLoader` creation.

6. Dictionary-style batch output containing images, labels, image paths, and image identifiers.

### Implementation

The `APTOSDataset` class was updated in:

```text

src/uncertainty_retfound/data/aptos.py
```

## Experiment: Baseline Training Infrastructure Smoke Test

**Date:** 2026-06-22  
**Status:** Completed  
**Related commits:**
- `f81656e` Add baseline model training smoke test
- `69c8984` Add reusable training epoch loop
- `0aef3a2` Add reusable evaluation loop
- `ea16c76` Add baseline classification metrics

### Objective

Validate the first end-to-end baseline training and evaluation infrastructure for the project using fake image data.

This milestone proves that the repository can move from image metadata and image files into model training, evaluation, and basic classification metrics before using the real APTOS dataset or integrating RETFound.

### Scope

The implemented baseline infrastructure supports:

1. A minimal CNN classifier.
2. One-batch training smoke tests.
3. Reusable one-epoch training loop.
4. Reusable evaluation loop.
5. Basic classification metrics.
6. Optional metric calculation during evaluation.
7. CPU-only fixture-based tests using fake PNG images.

### Implementation

The baseline model was added in:

```text
src/uncertainty_retfound/models/baseline.py
```
## Experiment Infrastructure: Baseline Training CLI

**Date:** 2026-06-22  
**Status:** Completed  
**Related commit:**
- `a2ebf1e` Add baseline training CLI

### Objective

Add a reproducible command-line entry point for running the baseline training pipeline from prepared metadata and an image root directory.

This milestone connects the existing dataset, dataloader, baseline model, training loop, evaluation loop, and classification metrics into a single runnable command.

### Scope

The CLI supports:

    loading prepared metadata CSV
    filtering train and validation splits
    constructing torchvision transforms
    creating APTOSDataset instances
    creating PyTorch DataLoaders
    training SmallCNNClassifier
    evaluating after each epoch
    writing metrics.json to an output directory

### Validation

The CLI is covered by tests using fake PNG images under temporary directories.

## Experiment: RETFound Feature Export and Cached-Feature Linear Head Baseline

**Date:** 2026-06-24  
**Status:** Completed

### Objective

Complete the first frozen-RETFound feature workflow on APTOS 2019 and establish a cached-feature softmax linear-head baseline plus post-hoc temperature scaling.

This milestone moves the project from image-based smoke infrastructure to a reusable feature-based experiment path that is appropriate for the planned Bayesian and Laplace last-layer comparisons.

### Feature Export Milestone

Frozen RETFound features were exported successfully to:

```text
outputs/features/aptos2019_retfound_mae_natureCFP/
```

Cached feature shapes:

1. `train`: `(2930, 1024)`
2. `val`: `(366, 1024)`
3. `test`: `(366, 1024)`

This confirms that the external RETFound adapter, APTOS metadata pipeline, preprocessing path, and split-aware export script now work together on the real dataset.

### Cached-Feature Softmax Linear Head

A 5-epoch cached-feature softmax linear-head run with batch size 8 completed at:

```text
outputs/feature_heads/retfound_softmax_linear_5epoch_bs8
```

Final validation results:

1. Accuracy: `0.8798`
2. AUC: `0.9580`
3. Sensitivity: `0.8896`
4. Specificity: `0.8726`
5. Negative log likelihood: `0.2643`
6. Brier score: `0.0807`
7. Expected calibration error: `0.0348`
8. Confusion matrix: `[[185, 27], [17, 137]]`

### Temperature Scaling

Post-training temperature scaling was fit on the cached-feature validation predictions.

Result:

1. Learned temperature: `0.7649`
2. ECE improved from `0.0348` to `0.0186`
3. NLL improved from `0.2643` to `0.2558`
4. Brier score improved from `0.0807` to `0.0798`
5. Classification metrics were unchanged, as expected for temperature scaling

### Interpretation

The cached-feature baseline is comparable to the image-based baseline, but it should not be treated as strictly identical. The remaining gap is likely due to optimization differences and the fact that the cached-feature path uses fixed exported features rather than updating image-batch representations during training.

An additional practical observation is that batch size 32 undertrained relative to batch size 8 because it produced fewer optimizer updates over the same epoch budget.

### Consequence For Next Experiments

All future Bayesian-head and Laplace-head comparisons should use:

1. The cached-feature softmax linear baseline
2. The cached-feature temperature-scaled baseline

as the primary deterministic reference points.

## Experiment: First Variational Bayesian Cached-Feature Head

**Date:** 2026-06-24  
**Status:** Completed

### Objective

Run the first variational Bayesian linear head on cached RETFound features and evaluate whether it improves on the cached softmax deterministic baseline while also producing uncertainty signals that separate correct from incorrect predictions.

### Run Configuration

Output path:

```text
outputs/feature_heads/retfound_variational_bayesian_20epoch_best_val_loss
```

Model:

1. Variational Bayesian linear head on cached RETFound features

Training setup:

1. Epochs: `20`
2. Batch size: `8`
3. Learning rate: `0.001`
4. `mc_samples_train=1`
5. `mc_samples_eval=30`
6. `prior_std=1.0`
7. `kl_weight=1/2930`
8. `selection_metric=val_loss`

### Best Epoch Result

Best epoch: `16`

Best validation metrics:

1. Accuracy: `0.8934`
2. AUC: `0.9650`
3. Sensitivity: `0.8831`
4. Specificity: `0.9009`
5. Balanced accuracy: `0.8920`
6. ECE: `0.0216`
7. NLL: `0.2371`
8. Brier score: `0.0724`
9. Confusion matrix: `[[191, 21], [18, 136]]`

### Comparison Against Cached Softmax + Temperature Scaling

Relative to the cached softmax baseline with temperature scaling, the variational Bayesian head improved:

1. Accuracy
2. AUC
3. NLL
4. Brier score
5. Specificity

At the same time:

1. ECE was slightly worse
2. Sensitivity was slightly lower

This is encouraging, but it is not enough to claim that the Bayesian head is uniformly better. The result is stronger on several aggregate metrics, but the calibration and error profile still need closer analysis.

### Uncertainty Separation

The uncertainty summaries show meaningful separation between correct and incorrect predictions:

1. Mean confidence: correct `0.9084` vs incorrect `0.6929`
2. Mean predictive entropy: correct `0.2375` vs incorrect `0.5682`
3. Mean probability variance: correct `0.0064` vs incorrect `0.0235`
4. Mean mutual information: correct `0.0224` vs incorrect `0.0572`

These are useful signs that the Bayesian head is capturing epistemic and predictive uncertainty in a direction consistent with error detection, but they should still be validated with selective-referral and thresholded error analyses.

### Clinical Safety Caveat

One high-confidence false negative remains in the current run. That means the uncertainty summaries are promising, but not sufficient on their own. Selective-referral analysis and explicit high-confidence-error analysis remain necessary before drawing stronger safety conclusions.

### Next Step

The next planned step is either:

1. Compare alternative best-epoch selection metrics
2. Implement the Laplace last-layer baseline

## Experiment: Sensitivity-Selected Variational Bayesian Cached-Feature Head

**Date:** 2026-06-24  
**Status:** Completed

### Objective

Evaluate whether selecting the variational Bayesian cached-feature model by sensitivity produces a more screening-oriented operating point than the earlier val-loss-selected Bayesian run and the cached softmax deterministic baseline.

### Run Configuration

Output path:

```text
outputs/feature_heads/retfound_variational_bayesian_20epoch_best_sensitivity
```

Model:

1. Variational Bayesian linear head on cached RETFound features

Training setup:

1. Epochs: `20`
2. Batch size: `8`
3. Learning rate: `0.001`
4. `mc_samples_train=1`
5. `mc_samples_eval=30`
6. `prior_std=1.0`
7. `kl_weight=1/2930`
8. `selection_metric=sensitivity`

### Best Epoch Result

Best epoch: `7`

Best validation metrics:

1. Accuracy: `0.8962`
2. AUC: `0.9617`
3. Sensitivity: `0.9545`
4. Specificity: `0.8538`
5. Balanced accuracy: `0.9042`
6. ECE: `0.0305`
7. NLL: `0.2710`
8. Brier score: `0.0816`
9. Confusion matrix: `[[181, 31], [7, 147]]`

### Comparison Against Cached Softmax + Temperature Scaling

Relative to the cached softmax baseline with temperature scaling:

1. Sensitivity improved from `0.8896` to `0.9545`
2. False negatives decreased from `17` to `7`
3. Accuracy improved
4. AUC improved
5. ECE, NLL, and Brier score were worse

This makes the run interesting as a screening-oriented Bayesian operating point, but not a uniformly better calibrated model. The sensitivity gain came with weaker calibration-oriented summary metrics and lower specificity than the val-loss-selected Bayesian result.

### Uncertainty Separation

The uncertainty summaries again separated correct from incorrect predictions:

1. Mean confidence: correct `0.8883` vs incorrect `0.7248`
2. Mean predictive entropy: correct `0.2789` vs incorrect `0.5376`
3. Mean probability variance: correct `0.0021` vs incorrect `0.0053`
4. Mean mutual information: correct `0.0067` vs incorrect `0.0130`

This is directionally useful, but it still does not remove the need for explicit referral-threshold and error-slice analysis.

### Persistent Hard False Negative

The hard false-negative case `id_code 025a169a0bb0` persists in this run. Its confidence was reduced to `0.8739`, compared with `0.9291` in the val-loss-selected Bayesian model and `0.9720` in the final epoch model. That reduction is encouraging, but the case still remains a false negative and should be treated as an unresolved safety concern rather than a solved error mode.

### Interpretation

This result gives a plausible screening-oriented Bayesian operating point: much higher sensitivity and fewer false negatives, with a cost in calibration metrics and some loss of specificity. It is promising, but it should not be overclaimed as a definitive clinically safer model without selective-referral analysis and targeted review of high-confidence errors.

## Experiment: Bayesian Hyperparameter Sweep For Sensitivity-Selected Cached-Feature Heads

**Date:** 2026-06-24  
**Status:** Completed

### Objective

Measure how Bayesian hyperparameters change the sensitivity-specificity-calibration-uncertainty tradeoff for cached-feature variational Bayesian heads when model selection is based on sensitivity.

### Sweep Setup

Sweep location:

```text
outputs/feature_heads/sweeps/
```

The sweep varied:

1. `kl_weight` in `[0.00003, 0.0001, 0.000341, 0.001]`
2. `prior_std` in `[0.5, 1.0, 2.0]`

All runs used:

1. `selection_metric=sensitivity`
2. `20` epochs
3. Batch size `8`
4. Learning rate `0.001`
5. `mc_samples_train=1`
6. `mc_samples_eval=30`

### Maximum-Sensitivity Candidate

Output path:

```text
outputs/feature_heads/sweeps/bayes_sensitivity_kl_0.001_prior_2.0
```

Best epoch: `7`

Validation metrics:

1. Accuracy: `0.8934`
2. AUC: `0.9613`
3. Sensitivity: `0.9610`
4. Specificity: `0.8443`
5. Balanced accuracy: `0.9027`
6. ECE: `0.0320`
7. NLL: `0.2794`
8. Brier score: `0.0842`
9. Confusion matrix: `[[179, 33], [6, 148]]`

Uncertainty separation:

1. Mean confidence: correct `0.8853` vs incorrect `0.7253`
2. Mean predictive entropy: correct `0.2841` vs incorrect `0.5356`
3. Mean probability variance: correct `0.0055` vs incorrect `0.0139`
4. Mean mutual information: correct `0.0179` vs incorrect `0.0344`

The persistent hard false-negative case `025a169a0bb0` remained incorrect, but its confidence decreased further to `0.8498`.

### Balanced Screening Candidate

Output path:

```text
outputs/feature_heads/sweeps/bayes_sensitivity_kl_0.00003_prior_2.0
```

Best epoch: `7`

Validation metrics:

1. Accuracy: `0.9016`
2. AUC: `0.9617`
3. Sensitivity: `0.9545`
4. Specificity: `0.8632`
5. Balanced accuracy: `0.9089`
6. ECE: `0.0275`
7. NLL: `0.2639`
8. Brier score: `0.0796`
9. Confusion matrix: `[[183, 29], [7, 147]]`

This candidate gives a better overall balance than the maximum-sensitivity setting, but the posterior uncertainty signal is weaker.

### Comparison Against Cached Softmax + Temperature Scaling

The cached softmax plus temperature-scaled baseline had sensitivity `0.8896` with `17` false negatives.

Relative to that baseline:

1. The maximum-sensitivity Bayesian candidate reduced false negatives from `17` to `6`
2. The balanced Bayesian candidate reduced false negatives from `17` to `7`
3. The balanced Bayesian candidate also improved accuracy and balanced accuracy

### Interpretation

The sweep suggests that Bayesian hyperparameters materially affect the sensitivity-specificity-uncertainty tradeoff. That is useful for the project, because it means the Bayesian head can be tuned toward different operating goals. It does not mean that one setting dominates all others; the tradeoff remains real and should be described explicitly.

## Experiment: Laplace Last-Layer Baseline And Prior-Precision Sweep

**Date:** 2026-06-24  
**Status:** Completed

### Objective

Establish a Laplace last-layer baseline on cached RETFound features and test whether prior precision changes the screening-oriented tradeoff in a useful way.

### Implementation Summary

The Laplace baseline uses:

1. A deterministic cached-feature linear head trained first
2. A diagonal Laplace posterior approximation fitted afterward over the final layer

This gives a lightweight Bayesian last-layer baseline without variational training.

### Initial Laplace Val-Loss-Selected Run

Output path:

```text
outputs/feature_heads/retfound_laplace_20epoch_best_val_loss
```

Best epoch: `15`

Validation metrics:

1. Accuracy: `0.8962`
2. AUC: `0.9666`
3. Sensitivity: `0.8636`
4. Specificity: `0.9198`
5. Balanced accuracy: `0.8917`
6. ECE: `0.0972`
7. NLL: `0.2891`
8. Brier score: `0.0826`
9. Confusion matrix: `[[195, 17], [21, 133]]`

### Initial Laplace Sensitivity-Selected Run

Output path:

```text
outputs/feature_heads/retfound_laplace_20epoch_best_sensitivity
```

Best epoch: `6`

Validation metrics:

1. Accuracy: `0.8962`
2. AUC: `0.9574`
3. Sensitivity: `0.9416`
4. Specificity: `0.8632`
5. Balanced accuracy: `0.9024`
6. ECE: `0.1231`
7. NLL: `0.3349`
8. Brier score: `0.0971`
9. Confusion matrix: `[[183, 29], [9, 145]]`

### Prior-Precision Sweep

Sweep location:

```text
outputs/feature_heads/sweeps/laplace_sensitivity_prior_precision_*/
```

Sweep values:

1. `prior_precision` in `[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]`

### Best-Sensitivity Laplace Candidate

Best sensitivity occurred at:

1. `prior_precision=0.03`
2. Best epoch `6`

Validation metrics:

1. Sensitivity: `0.9481`
2. Specificity: `0.8632`
3. Accuracy: `0.8989`
4. AUC: `0.9585`
5. Balanced accuracy: `0.9056`
6. ECE: `0.1382`
7. NLL: `0.3454`
8. Brier score: `0.1002`
9. Confusion matrix: `[[183, 29], [8, 146]]`

### Better-Calibrated Laplace Sweep Candidate

A somewhat better-calibrated Laplace sweep candidate occurred at:

1. `prior_precision=10.0`
2. Best epoch `6`

Validation metrics:

1. Sensitivity: `0.9481`
2. Specificity: `0.8585`
3. Accuracy: `0.8962`
4. AUC: `0.9586`
5. Balanced accuracy: `0.9033`
6. ECE: `0.0900`
7. NLL: `0.3061`
8. Brier score: `0.0894`
9. Confusion matrix: `[[182, 30], [8, 146]]`

### Comparison Against Cached Softmax + Temperature Scaling

The cached softmax plus temperature-scaled baseline had `17` false negatives.

Compared with that baseline:

1. The best sensitivity Laplace setting reduced false negatives from `17` to `8`
2. ECE, NLL, and Brier score were worse

### Comparison Against Variational Bayesian

Relative to the variational Bayesian cached-feature results:

1. Variational Bayesian remained stronger on the sensitivity-balanced-screening tradeoff
2. Variational Bayesian achieved `6` or `7` false negatives depending on operating point
3. The diagonal Laplace approximation did not match variational Bayesian on calibration

### Interpretation

The Laplace baseline is useful as a Bayesian uncertainty reference and gives a meaningful reduction in false negatives relative to the deterministic cached-feature baseline. However, this diagonal approximation did not outperform either variational Bayesian or temperature scaling on calibration-oriented metrics. It should therefore be treated as a credible baseline, not as the current preferred uncertainty method.

## Experiment: Clinical Decision-Policy Comparison

**Date:** 2026-06-24  
**Status:** Completed

### Objective

Compare simple clinical decision policies built from cached-feature model predictions to test whether straightforward ensemble referral rules improve safety beyond the best single-model screening operating points.

### Output Files

1. `outputs/decision_policies/retfound_policy_comparison.json`
2. `outputs/decision_policies/retfound_policy_comparison.csv`

### Models And Policies Compared

1. `softmax_temp`
2. `bayes_max_sensitivity`
3. `bayes_balanced`
4. `laplace_sensitivity`
5. `OR rule`
6. `majority vote`
7. `AND rule`

### Key Single-Model Results

1. `softmax_temp`: sensitivity `0.8896`, specificity `0.8726`, false negatives `17`, false positives `27`, referral_rate `0.4481`
2. `bayes_max_sensitivity`: sensitivity `0.9610`, specificity `0.8443`, false negatives `6`, false positives `33`, referral_rate `0.4945`
3. `bayes_balanced`: sensitivity `0.9545`, specificity `0.8632`, false negatives `7`, false positives `29`, referral_rate `0.4809`
4. `laplace_sensitivity`: sensitivity `0.9481`, specificity `0.8585`, false negatives `8`, false positives `30`, referral_rate `0.4809`

### Key Ensemble And Policy Results

1. `OR rule`: sensitivity `0.9610`, specificity `0.8302`, false negatives `6`, false positives `36`, referral_rate `0.5027`
2. `majority vote`: sensitivity `0.9545`, specificity `0.8491`, false negatives `7`, false positives `32`, referral_rate `0.4891`
3. `AND rule`: sensitivity `0.8896`, specificity `0.8915`, false negatives `17`, false positives `23`, referral_rate `0.4372`

### Interpretation

The ensemble rules did not improve on the strongest single-model Bayesian operating points.

1. The `OR rule` did not reduce false negatives below `bayes_max_sensitivity`; both had `6` false negatives
2. The `OR rule` increased false positives and referral rate relative to `bayes_max_sensitivity`
3. `majority vote` did not improve over `bayes_balanced`; it had the same false negatives but more false positives
4. The `AND rule` improved specificity but reduced sensitivity and is not suitable for screening-oriented referral
5. The remaining false-negative cases may be shared hard cases across models rather than isolated model-specific misses

### Conclusion

1. The best maximum-sensitivity operating point remains `bayes_max_sensitivity`
2. The best balanced screening operating point remains `bayes_balanced`
3. Simple ensemble referral policies did not improve safety beyond the best single Bayesian operating point

The tests verify:

    metrics.json is created
    expected JSON keys are present
    epoch metrics are recorded
    train loss is finite and non-negative
    validation loss is finite and non-negative
    validation accuracy is included
    missing split column raises a clear error
    empty train split raises a clear error
    empty validation split raises a clear error

The full test suite passed with:

    53 passed

### Notes

The CLI does not download data and does not require the full APTOS dataset for tests.

For real experiments, the CLI should be run on the cluster after:

    full APTOS dataset download
    metadata preparation
    image path validation with zero missing images

The first cluster run should be treated as a baseline smoke experiment before RETFound integration.

### Next Steps

    run full APTOS metadata preparation on the cluster
    validate all real image paths on the cluster
    run a one-epoch baseline smoke experiment
    record the first real baseline metrics
    integrate RETFound after the baseline path is validated

## Experiment: Full APTOS Baseline Smoke Run on Cluster

**Date:** 2026-06-23  
**Status:** Completed  
**Environment:** Cluster GPU instance  
**GPU:** NVIDIA A100 40GB  
**Task:** Referable diabetic retinopathy classification  
**Model:** SmallCNNClassifier  
**Dataset:** APTOS 2019 mirror from Kaggle  
**Output directory:** `outputs/baseline_referable_dr_smoke_progress`

### Objective

Validate the full real-data training pipeline on the cluster before integrating RETFound.

This smoke run was intended to confirm that the project can:

    download and store the full APTOS dataset on the cluster
    prepare referable DR metadata
    validate real image paths
    create a unified image root
    train on GPU
    evaluate on the validation split
    save epoch-level metrics
    save batch-level training and validation history

This run was not intended to produce a strong model.

### Dataset Preparation

The Kaggle mirror provided separate metadata files and image folders:

    train_1.csv
    valid.csv
    test.csv

The split sizes were:

    train: 2930
    val: 366
    test: 366
    total: 3662

For the referable DR binary task, labels were mapped as:

    diagnosis 0 or 1 -> label 0
    diagnosis 2, 3, or 4 -> label 1

The resulting label distribution was:

    label 0: 2175
    label 1: 1487

Validation split label distribution:

    label 0: 212
    label 1: 154

The majority-class validation baseline is therefore:

    212 / 366 = 0.5792

### Image Validation

The dataset mirror stores images in split-specific folders:

    data/raw/aptos2019/train_images/train_images
    data/raw/aptos2019/val_images/val_images
    data/raw/aptos2019/test_images/test_images

Image path validation passed for all splits:

    train: rows=2930, missing=0
    val: rows=366, missing=0
    test: rows=366, missing=0

A unified symlink image root was created for the training CLI:

    data/raw/aptos2019/all_images

The unified image root contained:

    3662 symlink entries
    3662 valid resolved image files

### Command

The smoke run used:

    uv run python -m scripts.training.train_baseline \
      --metadata-csv data/processed/aptos2019_referable_dr_metadata_splits.csv \
      --image-root data/raw/aptos2019/all_images \
      --output-dir outputs/baseline_referable_dr_smoke_progress \
      --num-classes 2 \
      --epochs 1 \
      --batch-size 32 \
      --learning-rate 0.001 \
      --resize 256 \
      --center-crop 224 \
      --device cuda

### Results

The one-epoch smoke run completed successfully on GPU.

Epoch-level metrics:

    train_loss: 0.6761
    val_loss: 0.6787
    val_accuracy: 0.5792

Validation confusion matrix:

    [[212,   0],
     [154,   0]]

Per-class validation accuracy:

    class 0: 1.0
    class 1: 0.0

Batch history was saved successfully:

    train batch records: 92
    validation batch records: 12

### Interpretation

The model predicted only the majority class on the validation set.

The validation accuracy therefore matches the majority-class baseline:

    0.5792

This confirms that the tiny CNN did not learn meaningful disease signal in the one-epoch smoke run. That is acceptable because the purpose of this experiment was infrastructure validation, not model performance.

The smoke run successfully validated the real-data training path:

    full APTOS data available on cluster
    metadata preparation works
    image path validation works
    unified image root works
    GPU training works
    metrics.json is written
    batch-level history is written
    validation metrics are interpretable

### Next Step

Move from the tiny CNN infrastructure baseline to the real project baseline:

    RETFound frozen encoder
    simple linear classification head
    referable DR task
    validation metrics
    later calibration and uncertainty metrics

The tiny CNN baseline should be treated as an infrastructure smoke baseline, not the main model baseline for the research project.

## Model Infrastructure: Staged RETFound Linear Baseline Interface

**Date:** 2026-06-23  
**Status:** Completed  
**Related commit:**
- `759b8ef` Add staged RETFound linear baseline interface

### Objective

Add a tested model boundary for the future RETFound frozen-encoder baseline without requiring real RETFound weights during local development or CI.

The goal was to prepare the codebase for the real project baseline:

    RETFound frozen encoder
    trainable linear classification head
    referable DR classification

### Scope

This milestone added:

    FrozenEncoderClassifier
    RETFound-style checkpoint-path entry point
    model selection in the baseline training CLI
    CLI support for retfound_linear
    tests using fake encoders

### Implemented Behavior

The `FrozenEncoderClassifier` supports several common encoder output formats:

    2D tensor features
    3D token features using the CLS token
    dictionaries with features
    dictionaries with pooler_output
    dictionaries with last_hidden_state
    tuple/list outputs using the first element

The classifier freezes encoder parameters by default and adds a trainable linear head.

The staged RETFound builder validates that a checkpoint path exists, but intentionally raises `NotImplementedError` for real RETFound architecture loading.

### Validation

The full test suite passed with:

    66 passed

The tests cover:

    output shape
    frozen encoder behavior
    unfrozen encoder behavior
    supported encoder output formats
    unsupported output errors
    feature-dimension mismatch errors
    CLI backward compatibility with small_cnn
    clear failure behavior for staged retfound_linear path

### Notes

This is a staged interface milestone, not a completed RETFound experiment.

The project still does not load actual RETFound weights.

Real RETFound support still requires adding a compatible architecture/checkpoint-loading path.

### Next Steps

    decide on RETFound architecture-loading strategy
    add real RETFound checkpoint loading support
    keep checkpoint files outside Git
    run RETFound frozen-encoder smoke experiment on the cluster
    compare RETFound deterministic baseline against later uncertainty-aware methods

## First real RETFound linear baseline run

This run records the first successful real RETFound baseline on the APTOS 2019 referable diabetic retinopathy task. Unlike the earlier small CNN smoke test, this experiment used a frozen RETFound-MAE encoder with a trainable linear classification head.

### External RETFound setup

- External RETFound repo path on cluster: `/home/karim/external/RETFound_MAE`
- External RETFound repo remote: `https://github.com/rmaphoh/RETFound_MAE.git`
- External RETFound repo commit: `ae9a9ecf37857cf47b8aa9f87cd6f710d75db287`
- Hugging Face checkpoint: `YukunZhou/RETFound_mae_natureCFP`
- Local checkpoint path on cluster: `/home/karim/models/retfound/RETFound_mae_natureCFP/RETFound_mae_natureCFP.pth`
- Architecture: `RETFound_mae`
- Feature dimension: `1024`
- Frozen encoder trainable parameters: `2,050`
- Total parameters: `303,303,682`

### Command

    uv run python -m scripts.training.train_baseline \
      --model-type retfound_linear \
      --retfound-repo-path /home/karim/external/RETFound_MAE \
      --backbone-checkpoint /home/karim/models/retfound/RETFound_mae_natureCFP/RETFound_mae_natureCFP.pth \
      --feature-dim 1024 \
      --metadata-csv data/processed/aptos2019_referable_dr_metadata_splits.csv \
      --image-root data/raw/aptos2019/all_images \
      --output-dir outputs/retfound_linear_referable_dr_smoke \
      --num-classes 2 \
      --epochs 1 \
      --batch-size 8 \
      --learning-rate 0.001 \
      --resize 224 \
      --center-crop 224 \
      --device cuda

### Results

- Train loss: `0.42631749279873365`
- Validation loss: `0.3518964662903645`
- Validation accuracy: `0.8360655903816223`
- Confusion matrix: `[[165, 47], [13, 141]]`
- Per-class accuracy:
  - Class 0: `0.7783018867924528`
  - Class 1: `0.9155844155844156`
- Train batches: `367`
- Validation batches: `46`

### Interpretation

This is the first successful real RETFound baseline for the project. It confirms that the external RETFound repository adapter, gated Hugging Face checkpoint loading, frozen feature extraction path, and linear classification head all work end to end on the cluster.

The run significantly improves over the earlier small CNN smoke baseline, which achieved approximately `0.5792` validation accuracy and predicted only class 0. The frozen RETFound encoder plus linear head detects both classes and achieves strong referable DR class accuracy.

This result should be treated as a successful integration and baseline milestone, not as final model performance. Accuracy is useful, but future research-quality evaluation should include sensitivity/recall, specificity, AUC, F1, calibration metrics, and uncertainty metrics.

### Next planned experiment

Run a 5-epoch frozen RETFound linear baseline using the same setup to check whether validation performance improves beyond the 1-epoch baseline while preserving strong referable DR sensitivity.

## Five-epoch frozen RETFound linear baseline

This experiment extends the first RETFound linear smoke run to five epochs using the same frozen RETFound-MAE encoder and trainable linear classification head.

### Configuration

- Model type: `retfound_linear`
- External RETFound repo path on cluster: `/home/karim/external/RETFound_MAE`
- Checkpoint: `/home/karim/models/retfound/RETFound_mae_natureCFP/RETFound_mae_natureCFP.pth`
- Architecture: `RETFound_mae`
- Feature dimension: `1024`
- Dataset: APTOS 2019 referable diabetic retinopathy
- Train split: `2,930` examples
- Validation split: `366` examples
- Number of classes: `2`
- Epochs: `5`
- Batch size: `8`
- Learning rate: `0.001`
- Resize / center crop: `224`
- Device: `cuda`
- Output directory: `outputs/retfound_linear_referable_dr_5epoch`

### Results by epoch

| Epoch | Train loss | Validation loss | Validation accuracy | Confusion matrix | Class 0 accuracy | Class 1 accuracy |
|---:|---:|---:|---:|---|---:|---:|
| 1 | 0.4263 | 0.3519 | 0.8361 | `[[165, 47], [13, 141]]` | 0.7783 | 0.9156 |
| 2 | 0.3412 | 0.3088 | 0.8607 | `[[175, 37], [14, 140]]` | 0.8255 | 0.9091 |
| 3 | 0.3148 | 0.2908 | 0.8934 | `[[197, 15], [24, 130]]` | 0.9292 | 0.8442 |
| 4 | 0.2900 | 0.2802 | 0.8770 | `[[179, 33], [12, 142]]` | 0.8443 | 0.9221 |
| 5 | 0.2782 | 0.2752 | 0.8852 | `[[178, 34], [8, 146]]` | 0.8396 | 0.9481 |

### Final validation metrics

- Final validation accuracy: `0.8852459192276001`
- Final validation confusion matrix: `[[178, 34], [8, 146]]`
- Final class 0 accuracy: `0.839622641509434`
- Final class 1 accuracy: `0.948051948051948`
- Train batches: `1835`
- Validation batches: `230`

### Interpretation

The five-epoch frozen RETFound linear baseline reached a strong validation accuracy of approximately `88.52%` at the final epoch. The highest raw validation accuracy occurred at epoch 3 with approximately `89.34%`, but epoch 5 is more attractive for a medical screening setting because it substantially reduced false negatives for referable diabetic retinopathy.

At epoch 3, the model had `24` false negatives for class 1. By epoch 5, this dropped to `8` false negatives, while the overall accuracy remained high. This suggests the final model is better aligned with a screening-oriented objective, where missing referable disease is typically more concerning than producing additional false positives.

This run establishes a strong frozen foundation-model baseline. Future work should add threshold-aware metrics, sensitivity/recall, specificity, AUC, F1, calibration metrics, and uncertainty-aware evaluation.

## Five-epoch RETFound linear baseline with classification and calibration metrics

This run repeats the five-epoch frozen RETFound linear baseline after adding richer binary classification metrics, calibration metrics, and validation prediction export.

### Configuration

- Model type: `retfound_linear`
- External RETFound repo path on cluster: `/home/karim/external/RETFound_MAE`
- Checkpoint: `/home/karim/models/retfound/RETFound_mae_natureCFP/RETFound_mae_natureCFP.pth`
- Architecture: `RETFound_mae`
- Feature dimension: `1024`
- Dataset: APTOS 2019 referable diabetic retinopathy
- Train split: `2,930` examples
- Validation split: `366` examples
- Number of classes: `2`
- Epochs: `5`
- Batch size: `8`
- Learning rate: `0.001`
- Resize / center crop: `224`
- Device: `cuda`
- Output directory: `outputs/retfound_linear_referable_dr_5epoch_metrics_predictions`
- Validation prediction export: `validation_predictions.csv`

### Results by epoch

| Epoch | Train loss | Val loss | Accuracy | AUC | Precision | Recall / Sensitivity | Specificity | F1 | Balanced accuracy | Brier | NLL | ECE | Mean confidence | Mean positive probability |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4263 | 0.3519 | 0.8361 | 0.9301 | 0.7500 | 0.9156 | 0.7783 | 0.8246 | 0.8469 | 0.1099 | 0.3519 | 0.0493 | 0.8068 | 0.4593 |
| 2 | 0.3412 | 0.3088 | 0.8607 | 0.9455 | 0.7910 | 0.9091 | 0.8255 | 0.8459 | 0.8673 | 0.0931 | 0.3088 | 0.0565 | 0.8263 | 0.4411 |
| 3 | 0.3148 | 0.2908 | 0.8934 | 0.9538 | 0.8966 | 0.8442 | 0.9292 | 0.8696 | 0.8867 | 0.0890 | 0.2908 | 0.0601 | 0.8334 | 0.3841 |
| 4 | 0.2900 | 0.2802 | 0.8770 | 0.9565 | 0.8114 | 0.9221 | 0.8443 | 0.8632 | 0.8832 | 0.0840 | 0.2802 | 0.0542 | 0.8527 | 0.4608 |
| 5 | 0.2782 | 0.2752 | 0.8852 | 0.9575 | 0.8111 | 0.9481 | 0.8396 | 0.8743 | 0.8938 | 0.0842 | 0.2752 | 0.0213 | 0.8687 | 0.4676 |

### Final validation metrics

- Accuracy: `0.8852459192276001`
- AUC: `0.957485910316099`
- Precision: `0.8111111111111111`
- Recall / sensitivity: `0.948051948051948`
- Specificity: `0.839622641509434`
- F1: `0.874251497005988`
- Balanced accuracy: `0.893837294780691`
- Brier score: `0.08415580540895462`
- Negative log likelihood: `0.2751513719558716`
- Expected calibration error: `0.021337965798508274`
- Mean confidence: `0.8687123656272888`
- Mean positive-class probability: `0.46755489706993103`
- Confusion matrix: `[[178, 34], [8, 146]]`
- Number of validation examples: `366`
- Correct predictions: `324`
- Incorrect predictions: `42`

### Best epochs

- Highest accuracy: epoch 3, validation accuracy `0.8934`
- Lowest validation loss: epoch 5, validation loss `0.2752`
- Highest recall / sensitivity: epoch 5, sensitivity `0.9481`
- Highest balanced accuracy: epoch 5, balanced accuracy `0.8938`
- Lowest expected calibration error: epoch 5, ECE `0.0213`
- Highest AUC: epoch 5, AUC `0.9575`

### Interpretation

This run establishes the strongest current softmax linear-head baseline for the project. Although epoch 3 had the highest raw accuracy, epoch 5 is the preferred screening-oriented baseline because it achieved the best sensitivity, balanced accuracy, validation loss, AUC, and calibration error.

The final model made only `8` false-negative predictions for referable diabetic retinopathy, while maintaining high overall validation accuracy and strong AUC. The expected calibration error of approximately `0.0213` is already low, which means future Bayesian or probabilistic heads will need to improve calibration meaningfully without reducing diagnostic performance.

This run is now the main standard softmax-head baseline for future comparisons against temperature scaling and Bayesian last-layer methods.

### Validation predictions

The run produced `validation_predictions.csv` with one row per validation example and the following fields:

- `id_code`
- `image_path`
- `true_label`
- `predicted_label`
- `probability_class_0`
- `probability_class_1`
- `confidence`
- `is_correct`

This file enables threshold analysis, calibration curves, selective prediction analysis, and high-confidence error analysis.
