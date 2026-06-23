# Cluster Data Workflow

## Purpose

This project uses a local-development plus cluster-execution workflow.

The local machine is used for code development, fixture-based testing, and small smoke tests. The full APTOS 2019 dataset is not downloaded locally because of storage constraints. Full dataset preparation, validation, baseline training, and later RETFound experiments should run on a cluster or another machine with sufficient storage and compute.

## Data Policy

Real APTOS images and generated experiment outputs must not be committed to Git.

The following paths are intended for local or cluster-only artifacts and are ignored by Git:

    data/
    outputs/
    results/

The repository should contain code, configs, tests, fixture data, documentation, and research process logs.

The repository should not contain full APTOS images, full prepared metadata outputs, model checkpoints, large experiment artifacts, or cluster logs.

## Local Development Workflow

Local development should continue to use fixture data and fake image files created during tests.

The local test suite validates the structure of the pipeline without requiring the full APTOS dataset.

Current local tests cover:

    APTOS metadata loading
    task mapping
    split generation
    metadata preparation script
    image path validation script
    APTOSDataset
    Pillow transforms
    torchvision tensor transforms
    DataLoader batching
    baseline model forward pass
    one-batch training
    epoch-level training
    evaluation loop
    classification metrics

Run local tests with:

    uv run pytest

The local machine does not need the full APTOS dataset for normal development.

## Expected Cluster Data Layout

On the cluster, place the full APTOS 2019 dataset under:

    data/raw/aptos2019/

Expected layout:

    data/raw/aptos2019/
      train.csv
      test.csv
      sample_submission.csv
      train_images/
        *.png
      test_images/
        *.png

Prepared metadata outputs should be written under:

    data/processed/

Experiment outputs should be written under:

    outputs/

Model checkpoints and larger experiment artifacts should be written under:

    results/

These directories are intentionally ignored by Git.

## Full Dataset Preparation on Cluster

After the full APTOS dataset is available on the cluster, prepare metadata splits with:

    uv run python scripts/data/prepare_aptos_metadata.py \
      --config configs/datasets/aptos2019.yaml \
      --output data/processed/aptos2019_referable_dr_metadata_splits.csv \
      --task referable_dr

The expected full APTOS training metadata size is approximately:

    3662 rows

The exact row count should be confirmed from the downloaded train.csv.

## Image Path Validation on Cluster

After preparing metadata, validate that all image paths resolve correctly:

    uv run python scripts/data/validate_aptos_images.py \
      --config configs/datasets/aptos2019.yaml \
      --metadata-csv data/processed/aptos2019_referable_dr_metadata_splits.csv

The expected result is:

    missing image count: 0

If images are missing, check the image root in the config, image extension, id_code column, train_images directory, and prepared metadata path.

## Baseline Training on Cluster

Baseline training should be run only after:

1. Full metadata preparation succeeds.
2. Image path validation reports zero missing images.
3. The local test suite passes.

The baseline training CLI has not yet been added. Once implemented, it should use:

    data/processed/aptos2019_referable_dr_metadata_splits.csv
    data/raw/aptos2019/train_images/

The first real baseline run should be treated as a smoke experiment, not as a final model result.

## RETFound Integration

RETFound integration should happen only after the baseline training path is stable on the full APTOS dataset.

Before RETFound integration, the project should have:

    validated full APTOS metadata
    validated image paths
    baseline training CLI
    baseline evaluation outputs
    basic metrics
    documented baseline experiment result

## Reproducibility Notes

For each cluster experiment, record:

    git commit hash
    dataset path
    task mapping
    split configuration
    transform configuration
    model configuration
    training command
    evaluation command
    main metrics
    known issues

These should be summarized in:

    research_process/experiment_log.md

Major design decisions should be summarized in:

    research_process/decision_log.md

## Current Decision

Because the local MacBook has limited free storage, the project will not download the full APTOS 2019 dataset locally.

Local development will continue with fixture and fake-image tests. Full-data validation and training will run on the cluster.

## Baseline Training CLI

The project now includes a baseline training CLI:

    scripts/training/train_baseline.py

This script does not download data. It expects prepared metadata and image files to already exist.

On the cluster, after metadata preparation and image path validation succeed, run a baseline smoke experiment with:

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

The CLI will:

    load prepared metadata
    filter train and validation splits
    build torchvision transforms
    create APTOSDataset instances
    create PyTorch DataLoaders
    train SmallCNNClassifier
    evaluate after each epoch
    write metrics.json under the output directory

The first baseline run should be treated as a smoke test. It verifies that the full-data training path works on the cluster before RETFound integration begins.

Expected output location:

    outputs/baseline_referable_dr_smoke/metrics.json

The output directory is ignored by Git and should not be committed.
