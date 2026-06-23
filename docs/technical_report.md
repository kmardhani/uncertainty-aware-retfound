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