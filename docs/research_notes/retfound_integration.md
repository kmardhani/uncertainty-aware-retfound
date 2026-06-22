# RETFound Integration Note

## Purpose

This document defines the initial plan for integrating RETFound into the project.

The goal is to use RETFound as the retinal foundation model backbone and evaluate whether Bayesian last-layer adaptation improves calibration, uncertainty estimation, selective referral, and robustness under dataset shift.

## Integration Strategy

The initial strategy is to use RETFound as a **frozen feature extractor**.

In this setup:

1. Retinal images are preprocessed according to RETFound expectations.
2. RETFound is loaded with pretrained weights.
3. The backbone is kept frozen.
4. Embeddings are extracted for train, validation, and test images.
5. Deterministic and Bayesian last-layer classifiers are trained on the saved embeddings.

This separates the central research question from the additional complexity of full foundation-model fine-tuning.

## Initial Model Choice

The initial model choice should be a RETFound model compatible with color fundus photography, since the initial datasets are expected to be APTOS, EyePACS, Messidor, or IDRiD.

The first candidate is:

- RETFound MAE model for color fundus images

The official RETFound repository should be treated as the primary implementation reference.

## Alternative Model Sources

Possible model sources include:

1. Official RETFound GitHub repository
2. Official or project-linked Hugging Face checkpoints
3. Transformers-compatible community conversions

The project should prefer official sources for the main experiments.

Community conversions may be useful for rapid prototyping if they simplify model loading, but they should be documented clearly and not silently substituted for the official implementation.

## Initial Embedding Strategy

The first embedding strategy will use the final RETFound representation for each image.

Possible embedding choices include:

- CLS token embedding
- pooled patch-token embedding
- output from the model’s penultimate representation

The first implementation should use the simplest representation supported by the official code or checkpoint.

The selected embedding method must be recorded in the decision log once confirmed.

## Preprocessing Requirements

Preprocessing should follow RETFound defaults as closely as possible.

Items to confirm:

- input image size
- crop or resize strategy
- normalization mean and standard deviation
- color channel ordering
- expected image format
- fundus image border handling
- whether images should be center-cropped, resized, or padded

The preprocessing pipeline should be implemented once and reused across datasets where possible.

## Local and Cluster Usage

Local development will be performed on a MacBook Pro M3 Pro with 18 GB unified memory.

Local usage should be limited to:

- code development
- import tests
- model-loading smoke tests if feasible
- preprocessing tests on a few images
- small embedding extraction tests
- last-layer experiments on small saved embeddings

The A100/H100 cluster should be used for:

- full-dataset RETFound embedding extraction
- large-scale feature storage
- repeated experiments
- Bayesian sampling or posterior approximation
- partial or full fine-tuning extensions

## Expected Feature Output Structure

Saved embeddings should be stored outside Git.

Recommended structure:

```text
outputs/
  features/
    aptos2019/
      train_embeddings.pt
      val_embeddings.pt
      test_embeddings.pt
      train_metadata.csv
      val_metadata.csv
      test_metadata.csv
    eyepacs/
      train_embeddings.pt
      val_embeddings.pt
      test_embeddings.pt
      train_metadata.csv
      val_metadata.csv
      test_metadata.csv