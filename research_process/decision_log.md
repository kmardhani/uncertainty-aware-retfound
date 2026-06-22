# Decision Log

## Decision 001 — Start with Frozen RETFound Embeddings

**Date:** 2026-06-22

### Decision

The initial experimental strategy will use RETFound as a frozen feature extractor. A deterministic last-layer classifier and one or more Bayesian last-layer variants will be trained on top of the extracted RETFound embeddings.

### Rationale

Starting with frozen RETFound embeddings keeps the first version of the project focused, reproducible, and easier to debug. It separates the central research question — whether Bayesian last-layer adaptation improves calibration, uncertainty estimation, selective referral, and robustness — from the additional complexity of full foundation-model fine-tuning.

This approach also allows faster iteration. Once RETFound embeddings are extracted, deterministic and Bayesian last-layer experiments can be repeated without repeatedly running the full image backbone.

### Consequences

- The first experiments will focus on last-layer adaptation rather than full model fine-tuning.
- Fine-tuning RETFound remains an optional extension.
- Saved embeddings will become an important intermediate artifact.
- Dataset preprocessing and feature extraction must be carefully documented.

### Status

Accepted.