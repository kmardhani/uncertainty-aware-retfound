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
