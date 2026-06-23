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
