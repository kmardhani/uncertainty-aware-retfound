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
