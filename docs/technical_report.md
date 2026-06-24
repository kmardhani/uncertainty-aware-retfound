## Dataset Preparation Pipeline

The project now includes an initial metadata preparation pipeline for the APTOS 2019 diabetic retinopathy dataset. This pipeline is designed to create reproducible experiment metadata before model training or RETFound integration begins.

The preparation workflow is implemented in:

```text

scripts/data/prepare_aptos_metadata.py
```


## APTOS Image Path Validation

The project now includes lightweight image path validation for the APTOS 2019 dataset. This validation step checks whether metadata rows can be resolved to expected retinal image file paths before model training begins.

The purpose of this stage is to catch missing or misconfigured image paths early, before implementing the PyTorch Dataset class or integrating RETFound.

### Implementation

Image path functionality is implemented in:

```text

src/uncertainty_retfound/data/aptos.py
```

## APTOS Dataset Wrapper

The project now includes a lightweight dataset wrapper for loading APTOS image-label examples from prepared metadata.

The implementation is located in:

```text

src/uncertainty_retfound/data/aptos.py
```

## APTOS Image Preprocessing Transforms

The project now includes a small image preprocessing layer for APTOS retinal images. This layer is intentionally lightweight and uses Pillow only. It is designed to work with the existing `APTOSDataset` transform interface before PyTorch, torchvision, or RETFound integration is introduced.

The implementation is located in:

```text

src/uncertainty_retfound/data/transforms.py
```

## PyTorch DataLoader Integration

The project now includes PyTorch and torchvision support for converting APTOS image-label examples into tensor batches.

This milestone bridges the data preparation and dataset wrapper layers into the training ecosystem, without yet introducing a model or training loop.

### Implementation

`APTOSDataset` now inherits from:

```text

torch.utils.data.Dataset
```

## Baseline Training Infrastructure

The project now includes a minimal baseline training and evaluation infrastructure. This milestone verifies that the data pipeline can feed image tensors into a model, compute classification loss, update model parameters, run evaluation, and compute basic classification metrics.

This infrastructure is intentionally simple and is designed to reduce risk before integrating RETFound.

### Baseline Model

The baseline model is implemented in:

```text
src/uncertainty_retfound/models/baseline.py
```
## Baseline Training CLI

The project now includes a runnable baseline training command implemented in:

    scripts/training/train_baseline.py

The CLI connects the existing data and training infrastructure into a reproducible experiment entry point.

It uses:

    prepared metadata CSV
    image root directory
    APTOSDataset
    torchvision transforms
    PyTorch DataLoaders
    SmallCNNClassifier
    train_one_epoch
    evaluate_model
    classification_summary

The CLI accepts command-line arguments for metadata path, image root, output directory, number of classes, number of epochs, batch size, learning rate, resize size, center crop size, split names, image identifier column, label column, image extension, device, and random seed.

The CLI does not download the dataset. It assumes that metadata preparation and image path validation have already completed.

A typical cluster smoke command is:

    uv run python scripts/training/train_baseline.py \
      --metadata-csv data/processed/aptos2019_referable_dr_metadata_splits.csv \
      --image-root data/raw/aptos2019/train_images \
      --output-dir outputs/baseline_referable_dr_smoke \
      --num-classes 2 \
      --epochs 1 \
      --batch-size 16 \
      --learning-rate 0.001 \
      --resize 256 \
      --center-crop 224

The CLI writes a JSON metrics file to:

    metrics.json

inside the requested output directory.

The metrics file records the input paths, training configuration, per-epoch training loss, per-epoch validation loss, validation accuracy when available, and final validation metrics where practical.

This provides the first reproducible experiment command for the project. It is intentionally limited to the small CNN baseline and does not include RETFound, checkpointing, calibration metrics, uncertainty metrics, or experiment tracking frameworks yet.

## Staged RETFound Linear Baseline Interface

The project now includes a staged RETFound-style model boundary implemented in:

    src/uncertainty_retfound/models/retfound.py

This milestone does not yet load real RETFound weights. Instead, it establishes the tested interface that will support the real RETFound baseline in the next implementation phase.

### Implemented Components

The main implemented model abstraction is:

    FrozenEncoderClassifier

This model wraps an arbitrary encoder and adds a trainable linear classification head.

It supports:

    frozen encoder parameters by default
    optional unfrozen encoder parameters
    2D encoder feature outputs
    3D token outputs using the CLS token
    dictionary outputs with features
    dictionary outputs with pooler_output
    dictionary outputs with last_hidden_state
    tuple/list outputs where the first item contains features
    clear errors for unsupported encoder outputs
    clear errors for feature-dimension mismatches

This makes the classifier compatible with a RETFound-style encoder while still allowing tests to use small fake encoders.

### Checkpoint Boundary

The project also includes a staged checkpoint entry point:

    build_retfound_linear_classifier

This function validates that a local checkpoint path exists, but intentionally raises `NotImplementedError` for actual RETFound architecture loading.

This is deliberate. The project does not yet claim to support real RETFound checkpoint loading.

Actual RETFound support still requires adding a compatible ViT/MAE architecture path, such as:

    official RETFound/MAE architecture code
    compatible timm ViT backend
    compatible Hugging Face/timm conversion path

The implementation intentionally does not auto-download RETFound weights.

### CLI Integration

The baseline training CLI now supports model selection:

    --model-type small_cnn
    --model-type retfound_linear

The default remains:

    --model-type small_cnn

Therefore, the existing cluster baseline command remains backward compatible.

For the staged RETFound path, the CLI supports:

    --backbone-checkpoint
    --feature-dim
    --freeze-encoder
    --unfreeze-encoder

At this stage, `retfound_linear` validates checkpoint-path requirements and then fails clearly because real RETFound architecture loading is not implemented yet.

### Testing

The staged interface is covered by tests using fake encoders. These tests verify:

    output logits have the expected shape
    encoder freezing works
    encoder unfreezing works
    2D tensor encoder outputs work
    3D token encoder outputs work
    dictionary feature outputs work
    last_hidden_state outputs work
    tuple/list outputs work
    unsupported outputs raise clear errors
    CLI model-selection behavior remains backward compatible

The full test suite passed with:

    66 passed

### Interpretation

This milestone creates the model abstraction and CLI boundary needed for the real RETFound baseline, but it should not be described as a completed RETFound experiment.

Completed:

    frozen encoder + linear head abstraction
    model-selection support in the training CLI
    local checkpoint-path validation
    fake-encoder tests

Not completed yet:

    real RETFound architecture loading
    real RETFound checkpoint compatibility
    real RETFound training on APTOS
    RETFound calibration or uncertainty evaluation

### Next Step

The next technical milestone is to add actual RETFound architecture/checkpoint loading support.

The recommended implementation direction is to keep the checkpoint local and explicit:

    --backbone-checkpoint /path/to/retfound_weights.pth

This keeps large model weights out of Git and avoids hidden automatic downloads.

## RETFound linear baseline milestone

The project now supports a real frozen RETFound-MAE encoder with a project-owned linear classification head. The integration follows the external-repo adapter strategy: RETFound source code and checkpoint files are kept outside this repository, while this repository provides the experiment pipeline, dataset handling, evaluation logic, and documentation.

### External dependency boundary

The RETFound integration uses explicit local paths:

- `--retfound-repo-path`
- `--backbone-checkpoint`

The project does not vendor RETFound source code and does not auto-download weights. This keeps the repository lightweight and makes external model provenance explicit.

The first successful cluster setup used:

- External repo: `/home/karim/external/RETFound_MAE`
- External repo remote: `https://github.com/rmaphoh/RETFound_MAE.git`
- External repo commit: `ae9a9ecf37857cf47b8aa9f87cd6f710d75db287`

## Cached RETFound Feature Pipeline

The project now includes a completed frozen-RETFound feature export path for APTOS 2019 together with a cached-feature softmax linear-head training path.

The exported real-data feature cache is stored at:

```text
outputs/features/aptos2019_retfound_mae_natureCFP/
```

The cached split shapes are:

1. `train`: `(2930, 1024)`
2. `val`: `(366, 1024)`
3. `test`: `(366, 1024)`

This confirms that the local APTOS metadata pipeline, image preprocessing, external RETFound adapter, and feature export CLI now work end to end on the full split structure needed for downstream experiments.

### Cached-Feature Deterministic Baseline

The first completed cached-feature softmax linear-head run is:

```text
outputs/feature_heads/retfound_softmax_linear_5epoch_bs8
```

Final validation metrics:

1. Accuracy: `0.8798`
2. AUC: `0.9580`
3. Sensitivity: `0.8896`
4. Specificity: `0.8726`
5. Negative log likelihood: `0.2643`
6. Brier score: `0.0807`
7. Expected calibration error: `0.0348`
8. Confusion matrix: `[[185, 27], [17, 137]]`

This result is comparable to the earlier image-based baseline, but not identical. That difference is expected: the cached-feature setting changes the optimization path and fixes the encoder outputs in advance, so it should be treated as a closely related but distinct deterministic baseline.

### Temperature-Scaled Cached-Feature Baseline

Temperature scaling on the cached-feature validation predictions produced:

1. Learned temperature: `0.7649`
2. ECE improvement from `0.0348` to `0.0186`
3. NLL improvement from `0.2643` to `0.2558`
4. Brier score improvement from `0.0807` to `0.0798`
5. No change in classification metrics

This is the expected pattern for post-hoc temperature scaling: ranking-based classification metrics and the confusion matrix stay fixed, while probability calibration improves.

### Optimization Note

Batch size 32 undertrained relative to batch size 8 because it produced fewer optimizer updates at the same epoch count. For the cached-feature linear-head setting, epoch count alone is therefore not a sufficient fairness control when comparing different batch sizes.

### Baseline Policy For Next Comparisons

For all upcoming Bayesian-head and Laplace-head experiments, the primary deterministic comparison points should be:

1. The cached-feature softmax linear baseline
2. The cached-feature temperature-scaled baseline

This keeps the comparison aligned with the frozen-feature experimental design rather than mixing feature-based uncertainty methods against only image-based end-to-end baselines.

## First Variational Bayesian Cached-Feature Head Result

The first completed variational Bayesian linear-head run on cached RETFound features is:

```text
outputs/feature_heads/retfound_variational_bayesian_20epoch_best_val_loss
```

### Model And Training Setup

The model is a variational Bayesian linear head trained on cached RETFound features with:

1. `20` epochs
2. Batch size `8`
3. Learning rate `0.001`
4. `mc_samples_train=1`
5. `mc_samples_eval=30`
6. `prior_std=1.0`
7. `kl_weight=1/2930`
8. `selection_metric=val_loss`

The selected best epoch was:

1. Epoch `16`

### Best Validation Metrics

1. Accuracy: `0.8934`
2. AUC: `0.9650`
3. Sensitivity: `0.8831`
4. Specificity: `0.9009`
5. Balanced accuracy: `0.8920`
6. ECE: `0.0216`
7. NLL: `0.2371`
8. Brier score: `0.0724`
9. Confusion matrix: `[[191, 21], [18, 136]]`

### Comparison To The Cached Softmax + Temperature-Scaled Baseline

Relative to the cached softmax baseline after temperature scaling, this Bayesian run improved:

1. Accuracy
2. AUC
3. NLL
4. Brier score
5. Specificity

However:

1. ECE was slightly worse
2. Sensitivity was slightly lower

This should be interpreted as a strong first Bayesian result, not as proof that the Bayesian head dominates the deterministic baseline on every clinically relevant axis.

### Uncertainty Separation

The run also showed useful separation between correct and incorrect predictions:

1. Mean confidence: correct `0.9084` vs incorrect `0.6929`
2. Mean predictive entropy: correct `0.2375` vs incorrect `0.5682`
3. Mean probability variance: correct `0.0064` vs incorrect `0.0235`
4. Mean mutual information: correct `0.0224` vs incorrect `0.0572`

These trends are consistent with uncertainty becoming larger on errors, which is encouraging for later selective-referral analysis.

### Safety Caveat

One high-confidence false negative still remains in this run. That is an important clinical caveat. The current result supports deeper uncertainty analysis, but it does not justify overclaiming safety or reliability from aggregate metrics alone.

### Next Step

The next planned step is either:

1. Compare alternative best-epoch selection metrics
2. Implement a Laplace last-layer baseline

## Sensitivity-Selected Variational Bayesian Cached-Feature Result

The project now has a second completed variational Bayesian cached-feature run selected by sensitivity:

```text
outputs/feature_heads/retfound_variational_bayesian_20epoch_best_sensitivity
```

### Model And Training Setup

This run uses the same variational Bayesian linear-head setup as the prior Bayesian experiment:

1. `20` epochs
2. Batch size `8`
3. Learning rate `0.001`
4. `mc_samples_train=1`
5. `mc_samples_eval=30`
6. `prior_std=1.0`
7. `kl_weight=1/2930`

The change is:

1. `selection_metric=sensitivity`

Best epoch:

1. Epoch `7`

### Best Validation Metrics

1. Accuracy: `0.8962`
2. AUC: `0.9617`
3. Sensitivity: `0.9545`
4. Specificity: `0.8538`
5. Balanced accuracy: `0.9042`
6. ECE: `0.0305`
7. NLL: `0.2710`
8. Brier score: `0.0816`
9. Confusion matrix: `[[181, 31], [7, 147]]`

### Comparison To Cached Softmax + Temperature Scaling

Relative to the cached softmax baseline after temperature scaling:

1. Sensitivity improved from `0.8896` to `0.9545`
2. False negatives decreased from `17` to `7`
3. Accuracy improved
4. AUC improved
5. ECE, NLL, and Brier score were worse

This makes the run important as a screening-oriented Bayesian operating point rather than a general win on every metric.

### Uncertainty Separation

Correct and incorrect predictions again show uncertainty separation:

1. Mean confidence: correct `0.8883` vs incorrect `0.7248`
2. Mean predictive entropy: correct `0.2789` vs incorrect `0.5376`
3. Mean probability variance: correct `0.0021` vs incorrect `0.0053`
4. Mean mutual information: correct `0.0067` vs incorrect `0.0130`

These trends are useful, but they are still descriptive rather than sufficient for deployment-relevant conclusions.

### Persistent Hard False Negative

The persistent hard false-negative case `025a169a0bb0` remains present. Its confidence fell to `0.8739`, compared with `0.9291` in the val-loss-selected Bayesian model and `0.9720` in the final epoch model. This is a meaningful reduction, but not a resolution of the failure mode.

### Interpretation

This result is best described as a screening-oriented Bayesian operating point. It pushes sensitivity substantially higher and reduces false negatives, but it does so with weaker calibration-oriented summary metrics than the temperature-scaled deterministic baseline. It is therefore promising for further screening analysis, not a basis for overclaiming overall superiority.

## Bayesian Hyperparameter Sweep

The project now includes a small hyperparameter sweep over variational Bayesian cached-feature heads at:

```text
outputs/feature_heads/sweeps/
```

### Sweep Design

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

The highest-sensitivity candidate is:

```text
outputs/feature_heads/sweeps/bayes_sensitivity_kl_0.001_prior_2.0
```

Best epoch:

1. Epoch `7`

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

The persistent hard false-negative case `025a169a0bb0` still remains, but its confidence drops to `0.8498`.

### Balanced Screening Candidate

A more balanced screening candidate is:

```text
outputs/feature_heads/sweeps/bayes_sensitivity_kl_0.00003_prior_2.0
```

Best epoch:

1. Epoch `7`

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

This candidate preserves most of the sensitivity gain while improving overall balance, but it appears to have a weaker posterior uncertainty signal.

### Comparison To Cached Softmax + Temperature Scaling

The cached softmax plus temperature-scaled baseline had:

1. Sensitivity `0.8896`
2. False negatives `17`

Compared with that baseline:

1. The maximum-sensitivity Bayesian candidate reduced false negatives from `17` to `6`
2. The balanced Bayesian candidate reduced false negatives from `17` to `7`
3. The balanced Bayesian candidate also improved accuracy and balanced accuracy

### Interpretation

The sweep suggests that Bayesian hyperparameters meaningfully change the operating point. In practice, they affect the tradeoff among sensitivity, specificity, calibration summaries, and uncertainty behavior. That is useful evidence for the project, but it should not be overinterpreted as proof that Bayesian heads automatically outperform deterministic baselines in every regime.

## Laplace Last-Layer Baseline

The project now includes a Laplace last-layer baseline built on cached RETFound features.

### Implementation

The implementation uses:

1. A deterministic cached-feature linear head trained first
2. A diagonal Laplace posterior approximation fitted afterward around the trained final-layer parameters

This keeps the method small and reproducible while still giving a Bayesian uncertainty baseline for comparison against the variational Bayesian head.

### Initial Val-Loss-Selected Laplace Run

Output path:

```text
outputs/feature_heads/retfound_laplace_20epoch_best_val_loss
```

Best epoch:

1. Epoch `15`

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

### Initial Sensitivity-Selected Laplace Run

Output path:

```text
outputs/feature_heads/retfound_laplace_20epoch_best_sensitivity
```

Best epoch:

1. Epoch `6`

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

## Laplace Prior-Precision Sweep

Sweep location:

```text
outputs/feature_heads/sweeps/laplace_sensitivity_prior_precision_*/
```

Sweep values:

1. `prior_precision` in `[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]`

### Best Sensitivity Laplace Result

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

### Better-Calibrated Laplace Candidate

A better-calibrated sweep candidate occurred at:

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

### Comparison To Cached Softmax + Temperature Scaling

The cached softmax plus temperature-scaled baseline had sensitivity-oriented false negatives of `17`.

Compared with that baseline:

1. The best sensitivity Laplace setting reduced false negatives from `17` to `8`
2. ECE, NLL, and Brier score were worse

### Comparison To Variational Bayesian

Compared with the variational Bayesian cached-feature runs:

1. Variational Bayesian remained stronger for the sensitivity-balanced-screening tradeoff
2. Variational Bayesian reached `6` or `7` false negatives depending on operating point
3. The diagonal Laplace approximation did not match variational Bayesian on calibration

### Interpretation

The Laplace last-layer baseline is useful because it provides a second Bayesian reference point with a different approximation strategy. However, this diagonal version did not outperform temperature scaling or the variational Bayesian head on calibration-oriented metrics. It should therefore be treated as a worthwhile baseline rather than the current best-performing Bayesian method.
- Checkpoint: `YukunZhou/RETFound_mae_natureCFP`
- Local checkpoint: `/home/karim/models/retfound/RETFound_mae_natureCFP/RETFound_mae_natureCFP.pth`
- Architecture: `RETFound_mae`
- Feature dimension: `1024`

### First baseline result

A 1-epoch frozen RETFound linear baseline was run on the APTOS 2019 referable diabetic retinopathy task.

Configuration:

- Model type: `retfound_linear`
- Encoder: frozen RETFound-MAE
- Linear head trainable parameters: `2,050`
- Total parameters: `303,303,682`
- Dataset split: train/validation
- Train examples: `2,930`
- Validation examples: `366`
- Batch size: `8`
- Learning rate: `0.001`
- Image resize/crop: `224`
- Device: `cuda`

Results:

- Train loss: `0.42631749279873365`
- Validation loss: `0.3518964662903645`
- Validation accuracy: `0.8360655903816223`
- Confusion matrix: `[[165, 47], [13, 141]]`
- Class 0 per-class accuracy: `0.7783018867924528`
- Class 1 per-class accuracy: `0.9155844155844156`

This result establishes a credible foundation-model baseline and confirms that the project has moved beyond infrastructure smoke testing. The earlier small CNN smoke run reached approximately `0.5792` validation accuracy and predicted only class 0, while the RETFound baseline detects both classes and performs substantially better.

Future evaluation should not rely on accuracy alone. For medical screening and trustworthy retinal disease classification, the project should also track sensitivity/recall, specificity, AUC, F1, calibration error, Brier score, negative log likelihood, and uncertainty-aware metrics.

The next planned experiment is a 5-epoch frozen RETFound linear baseline using the same setup.

## Five-epoch RETFound linear baseline

A five-epoch frozen RETFound linear baseline was run after the initial one-epoch integration run. The purpose was to determine whether the frozen encoder plus linear head could produce a stronger baseline without unfreezing the foundation model.

### Final result

- Final validation accuracy: `0.8852459192276001`
- Final validation loss: `0.2752`
- Final confusion matrix: `[[178, 34], [8, 146]]`
- Final class 0 accuracy: `0.839622641509434`
- Final class 1 accuracy: `0.948051948051948`

The highest raw validation accuracy occurred at epoch 3:

- Epoch 3 validation accuracy: `0.8934`
- Epoch 3 confusion matrix: `[[197, 15], [24, 130]]`
- Epoch 3 class 1 accuracy: `0.8442`

Although epoch 3 had the highest overall accuracy, epoch 5 is more clinically relevant for a screening-oriented task because it reduced false negatives from `24` to `8` while maintaining high overall validation accuracy. This highlights why future evaluation should prioritize sensitivity/recall, specificity, AUC, calibration, and uncertainty metrics in addition to accuracy.

## Softmax linear-head baseline with calibration metrics

The current strongest baseline is a frozen RETFound-MAE encoder with a standard softmax linear classification head trained for five epochs on the APTOS 2019 referable diabetic retinopathy task.

### Final epoch metrics

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

### Baseline interpretation

This baseline is strong for a frozen foundation-model linear probe. The final epoch is preferred over the peak-accuracy epoch because it provides a better screening-oriented tradeoff: higher sensitivity, lower validation loss, better balanced accuracy, higher AUC, and lower calibration error.

The result also creates a clear comparison target for future uncertainty-aware methods. A Bayesian or probabilistic last-layer method should aim to preserve the diagnostic performance of this baseline while improving calibration and uncertainty behavior.

### Future comparison criteria

Future Bayesian or calibrated heads should be compared against this softmax baseline using:

- Accuracy
- AUC
- Recall / sensitivity
- Specificity
- F1
- Balanced accuracy
- Brier score
- Negative log likelihood
- Expected calibration error
- Mean confidence
- High-confidence error rate
- Uncertainty separation between correct and incorrect predictions
- Selective prediction behavior

The main research objective is not simply to increase accuracy. The objective is to improve probability quality and uncertainty awareness while maintaining comparable diagnostic performance.
