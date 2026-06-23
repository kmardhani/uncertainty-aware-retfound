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
