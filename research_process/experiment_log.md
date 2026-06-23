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