## Execution Environments

This project is designed to support two execution environments:

1. **Local development environment**
2. **GPU cluster environment**

The local environment will be used for code development, documentation, lightweight testing, experiment analysis, and small-scale debugging.

The GPU cluster environment will be used for compute-intensive experiments, including RETFound feature extraction, optional fine-tuning, Bayesian last-layer experiments, repeated uncertainty sampling, and dataset-shift evaluations.

## Local Development Environment

Local development will use:

- Python 3.11
- `uv` for dependency management
- CPU or local GPU if available
- Small sample datasets or synthetic test inputs
- Lightweight unit tests and smoke tests

The local environment is not expected to run full training experiments.

## GPU Cluster Environment

The project has access to a GPU cluster with NVIDIA A100 and H100 GPUs. This makes larger-scale retinal foundation model experiments more feasible.

Cluster usage may include:

- RETFound checkpoint loading
- Large-scale feature extraction
- Frozen-backbone classifier training
- Partial or full fine-tuning experiments
- Bayesian last-layer posterior approximation
- Monte Carlo prediction sampling
- Cross-dataset evaluation
- Image perturbation and robustness experiments

## Compute Strategy

The project will use a staged compute strategy:

### Stage 1 — Local smoke tests

Before launching cluster jobs, code should be tested locally on a small subset of data.

Goals:

- Validate imports
- Validate data loading
- Validate preprocessing
- Validate model forward pass
- Validate metric computation
- Validate experiment configuration

### Stage 2 — Cluster feature extraction

Use the cluster to extract RETFound embeddings for the selected dataset.

This allows later experiments to run faster because many last-layer methods can train directly on saved embeddings.

Expected outputs:

```text
outputs/
  features/
    dataset_name/
      train_embeddings.pt
      val_embeddings.pt
      test_embeddings.pt
      metadata.csv