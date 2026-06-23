# Cluster Data Workflow

## Purpose

This project uses a local-development plus cluster-execution workflow.

The local machine is used for code development, fixture-based testing, and small smoke tests. The full APTOS 2019 dataset is not downloaded locally because of storage constraints. Full dataset preparation, validation, baseline training, and later RETFound experiments should run on a cluster or other machine with sufficient storage and compute.

## Data Policy

Real APTOS images and generated experiment outputs must not be committed to Git.

The following paths are intended for local or cluster-only artifacts and are ignored by Git:

```text
data/
outputs/
results/
```