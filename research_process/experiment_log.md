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