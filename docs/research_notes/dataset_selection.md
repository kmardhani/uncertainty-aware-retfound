# Dataset Selection Note

## Purpose

This document evaluates candidate retinal image datasets for the project and defines the initial dataset strategy.

The goal is to select datasets that support the project’s main research question:

**Does Bayesian last-layer adaptation on top of RETFound improve calibration, uncertainty estimation, selective referral, and robustness to dataset shift compared with a standard deterministic classification head?**

The dataset strategy should balance:

- research credibility
- practical feasibility
- reproducibility
- compute requirements
- suitability for uncertainty and calibration evaluation
- suitability for dataset-shift testing

## Project Requirements

The selected dataset or datasets should support:

- retinal disease classification
- diabetic retinopathy detection or grading
- calibration evaluation
- uncertainty estimation
- selective referral or abstention experiments
- possible dataset-shift evaluation
- reproducible train, validation, and test splits
- public or clearly permitted research use

The project should preserve original labels where possible so that multiple task formulations can be evaluated later.

## Local and Cluster Compute Context

Local development will be performed on a **MacBook Pro M3 Pro with 18 GB unified memory**.

This local machine is suitable for:

- repository development
- documentation
- small data samples
- preprocessing smoke tests
- metric implementation
- calibration plots
- analysis of saved predictions
- training lightweight last-layer models on saved embeddings

The local machine is not intended to be the primary environment for full RETFound feature extraction or fine-tuning on large retinal image datasets such as EyePACS.

The project also has access to a GPU cluster with **A100 and H100 GPUs**.

The cluster should be used for:

- full-dataset image preprocessing
- RETFound embedding extraction
- large-scale training runs
- partial or full RETFound fine-tuning experiments
- Bayesian posterior approximation or sampling
- repeated uncertainty experiments
- cross-dataset robustness evaluation

## Dataset Strategy

The project will use a staged dataset strategy.

### Stage 1 — Pipeline Dataset

Use **APTOS 2019** or a small **EyePACS subset** to validate the full research pipeline.

The purpose of this stage is to confirm that the following components work end to end:

- data loading
- preprocessing
- RETFound model loading
- embedding extraction
- deterministic last-layer training
- Bayesian last-layer training
- calibration evaluation
- uncertainty scoring
- selective referral evaluation
- experiment logging

This stage should prioritize fast iteration over maximum dataset size.

### Stage 2 — Main Dataset

Use **full EyePACS** or a large resized EyePACS version as the main dataset if storage, access, and preprocessing are manageable.

The purpose of this stage is to produce stronger main experimental results using a larger and more diverse retinal image dataset.

A larger dataset can improve the credibility of the work by providing:

- more statistical power
- greater image variability
- broader disease representation
- better class coverage
- stronger evaluation of calibration and uncertainty behavior

However, dataset size alone does not make the project research-grade. The main research credibility comes from a rigorous experimental design, careful evaluation, reproducibility, and honest interpretation of results.

### Stage 3 — External Dataset-Shift Evaluation

Use **Messidor**, **Messidor-2**, **IDRiD**, or another compatible diabetic retinopathy dataset for external validation and dataset-shift evaluation.

The purpose of this stage is to evaluate whether uncertainty-aware RETFound adaptation behaves more reliably when the test distribution differs from the training distribution.

This is especially important because the project is focused on trustworthiness, uncertainty, and robustness rather than classification accuracy alone.

## Candidate Datasets

| Dataset | Task | Strengths | Limitations | Possible Role |
|---|---|---|---|---|
| APTOS 2019 | Diabetic retinopathy grading | Manageable size, common benchmark, five DR severity classes | Kaggle access required, class imbalance, smaller than EyePACS | First pipeline dataset |
| EyePACS / Kaggle DR | Diabetic retinopathy grading | Large dataset, widely used, stronger main experiment candidate | Large download, noisy labels, heavier storage and preprocessing requirements | Main dataset after pipeline validation |
| EyePACS subset | Diabetic retinopathy grading | Allows faster development while staying close to the larger main dataset | Subset results may be less statistically strong | Alternative Stage 1 dataset |
| Messidor / Messidor-2 | Diabetic retinopathy evaluation | Useful external dataset for shift testing | Access and label compatibility need review | External validation / shift dataset |
| IDRiD | DR grading and lesion-related retinal data | Rich retinal disease dataset | Smaller and potentially more complex | External validation or extension |
| ODIR | Multi-disease ocular classification | Broader disease labels beyond DR | Multi-label classification complexity | Later extension |

## Recommended Initial Dataset

The recommended first dataset is **APTOS 2019 Blindness Detection** or a small **EyePACS subset**.

Between these two options:

- **APTOS 2019** is likely better for a fast first pipeline because it is smaller and easier to manage.
- **EyePACS subset** is useful if the project wants the first pipeline to match the later main dataset more closely.

The recommended first choice is:

**APTOS 2019 for initial pipeline validation.**

## Rationale for Starting with APTOS

APTOS is a good first dataset because it is smaller and more manageable than full EyePACS while still supporting a clinically relevant retinal classification task: diabetic retinopathy severity grading.

It has five severity categories:

- No diabetic retinopathy
- Mild diabetic retinopathy
- Moderate diabetic retinopathy
- Severe diabetic retinopathy
- Proliferative diabetic retinopathy

This makes it suitable for multiple task formulations:

- binary diabetic retinopathy detection
- referable diabetic retinopathy detection
- five-class diabetic retinopathy severity grading

Starting with APTOS allows the project to validate the research workflow before scaling to EyePACS on the cluster.

## Rationale for Scaling to EyePACS

EyePACS is a stronger candidate for the main research dataset because it is larger and more diverse.

Using EyePACS can strengthen the project by supporting:

- more robust performance estimates
- more meaningful calibration analysis
- better uncertainty evaluation
- more reliable selective referral curves
- stronger evidence that results are not only due to a small benchmark

However, EyePACS should not be the first implementation target unless the full pipeline has already been validated on a smaller dataset or subset.

The recommended approach is:

1. Validate pipeline on APTOS or small EyePACS subset.
2. Extract RETFound embeddings for full EyePACS on the A100/H100 cluster.
3. Train deterministic and Bayesian last-layer models on saved embeddings.
4. Evaluate calibration, uncertainty, and selective referral.
5. Use another dataset for external shift evaluation.

## Initial Classification Tasks

The project may define three related classification tasks.

### Task A — Binary DR Detection

Classes:

- No DR
- Any DR

Possible mapping:

- No DR: severity grade 0
- Any DR: severity grades 1, 2, 3, 4

Purpose:

- fast first baseline
- easier calibration interpretation
- useful proof of pipeline
- simple binary uncertainty analysis

### Task B — Referable DR Detection

Classes:

- Non-referable DR
- Referable DR

Possible mapping:

- Non-referable: no DR and mild DR
- Referable: moderate DR, severe DR, and proliferative DR

Purpose:

- more clinically meaningful screening-style task
- directly aligned with selective referral experiments
- useful for studying whether uncertain cases should be deferred to human review

### Task C — Five-Class DR Grading

Classes:

- No DR
- Mild DR
- Moderate DR
- Severe DR
- Proliferative DR

Purpose:

- full severity classification
- more difficult calibration and uncertainty problem
- closer to original dataset labels
- useful later after the binary and referable pipelines are working

## Recommended Initial Task

The recommended initial task is:

**Task B — Referable DR Detection**

## Rationale for Referable DR Detection

Referable DR detection is a strong fit for this project because the project is focused on trustworthy decision support and selective referral.

In this setup, the model predicts whether a retinal image likely belongs to a lower-risk or referral-worthy category. Uncertainty estimates can then be used to identify cases where the model should avoid making a confident automated prediction and instead defer to human review.

This task connects naturally to the project’s central themes:

- calibration
- uncertainty estimation
- selective referral
- robustness under shift
- trustworthiness in medical AI

## Dataset-Shift Plan

The dataset-shift evaluation should be staged after the in-distribution pipeline is working.

Possible shift settings include:

### Cross-Dataset Shift

Train or tune on one dataset and evaluate on another.

Possible examples:

- train on APTOS, evaluate on Messidor
- train on EyePACS, evaluate on Messidor
- train on EyePACS, evaluate on IDRiD

This is the most important shift setting for the project because it tests whether uncertainty estimates remain useful when the source of the retinal images changes.

### Synthetic Image Quality Shift

Apply controlled perturbations to test images.

Possible perturbations:

- blur
- brightness changes
- contrast changes
- compression artifacts
- cropping variation
- noise

This allows controlled testing of how model confidence and uncertainty respond to degraded image quality.

### Class Distribution Shift

Evaluate the model under different class distributions.

Possible approaches:

- oversample rare severity classes in the test set
- undersample common classes
- compare calibration across disease severity groups

This helps test whether calibration and uncertainty degrade when the class balance changes.

## Initial Experimental Path

The recommended dataset path is:

```text
1. Use APTOS 2019 or a small EyePACS subset for pipeline validation.
2. Implement referable DR detection as the first classification task.
3. Extract frozen RETFound embeddings.
4. Train a deterministic last-layer baseline.
5. Train a Bayesian last-layer model.
6. Compare accuracy, AUROC, calibration, uncertainty, and selective referral.
7. Scale to full EyePACS on the A100/H100 cluster.
8. Use Messidor, IDRiD, or another compatible dataset for external shift evaluation.