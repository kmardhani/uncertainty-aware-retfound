# Project Plan

## Purpose

This document defines the implementation roadmap for the project.

The plan is intentionally staged so that the project produces useful results early, while leaving room for deeper experiments such as fine-tuning and dataset-shift evaluation.

## Current Project Status

Completed:

- Repository created
- `uv` project initialized
- Python pinned to 3.11
- Initial repository structure created
- Initial project charter completed
- Initial environment plan completed

Next major focus:

- Dataset selection
- RETFound integration strategy
- Baseline deterministic classifier
- Bayesian last-layer adaptation

## Phase 1 — Project Foundation

### Goal

Establish a clear, reproducible research project structure.

### Tasks

- Create repository structure
- Add initial documentation files
- Define project charter
- Define environment plan
- Define project plan
- Add research process logs
- Confirm Git workflow

### Deliverables

- `README.md`
- `project_charter.md`
- `docs/environment_plan.md`
- `docs/project_plan.md`
- `research_process/decision_log.md`
- `research_process/experiment_log.md`
- `research_process/literature_review.md`
- `research_process/ai_supervision_log.md`

### Status

In progress.

## Phase 2 — Literature Review and Method Selection

### Goal

Ground the project in relevant prior work before implementing experiments.

### Tasks

- Review RETFound paper and implementation details
- Review Bayesian last-layer adaptation methods
- Review calibration metrics
- Review uncertainty estimation methods
- Review selective prediction and referral literature
- Review retinal disease classification datasets
- Select the first Bayesian method to implement

### Key Questions

- Which RETFound checkpoint should be used?
- Should RETFound initially be frozen?
- Which dataset should be used first?
- Which Bayesian last-layer method is simplest and strongest for the first experiment?
- What metrics are required for a convincing uncertainty-aware evaluation?

### Deliverables

- Updated `research_process/literature_review.md`
- Updated `research_process/decision_log.md`
- Initial method selection note

## Phase 3 — Dataset Selection and Data Pipeline

### Goal

Select the first retinal classification dataset and build a reproducible data loading pipeline.

### Tasks

- Identify candidate public retinal datasets
- Review dataset licenses and access requirements
- Select primary dataset
- Define classification task
- Define train/validation/test split strategy
- Implement data loading
- Implement preprocessing
- Add small smoke-test dataset workflow if possible

### Candidate Dataset Criteria

The selected dataset should have:

- Clear research permissions
- Retinal fundus images or compatible retinal imaging modality
- Classification labels suitable for the project
- Enough samples for meaningful experiments
- Potential for dataset-shift evaluation, either internally or with another dataset

### Deliverables

- Dataset selection note
- Data loading scripts
- Preprocessing scripts
- Dataset split files
- Dataset documentation in the technical report

## Phase 4 — RETFound Integration

### Goal

Integrate RETFound as the foundation model backbone or feature extractor.

### Tasks

- Locate official RETFound code and checkpoints
- Document checkpoint source
- Confirm required preprocessing
- Implement model loading
- Run a forward-pass smoke test
- Extract embeddings for a small sample
- Extract embeddings for the full selected dataset on the cluster

### Initial Strategy

Start with RETFound as a frozen feature extractor.

This keeps the first experiments focused and computationally manageable while still using RETFound as the foundation model.

### Deliverables

- RETFound loading code
- Feature extraction script
- Saved embedding files
- Smoke-test result
- Updated environment notes if needed

## Phase 5 — Deterministic Baseline

### Goal

Train a standard deterministic classifier on RETFound embeddings.

### Tasks

- Implement linear or shallow MLP classification head
- Train with cross-entropy loss
- Evaluate predictive performance
- Evaluate calibration
- Save predictions and metrics
- Generate initial reliability diagram

### Metrics

Performance metrics:

- Accuracy
- Balanced accuracy
- F1 score
- AUROC, where appropriate

Calibration metrics:

- Expected Calibration Error
- Negative log-likelihood
- Brier score
- Reliability diagram

### Deliverables

- Baseline training script
- Baseline evaluation script
- Metrics output
- Baseline experiment log entry

## Phase 6 — Bayesian Last-Layer Adaptation

### Goal

Implement and evaluate an uncertainty-aware final-layer model on top of RETFound embeddings.

### Initial Candidate Methods

Candidate methods include:

- Bayesian logistic regression
- Laplace approximation over the final layer
- Monte Carlo posterior sampling
- Ensemble-style fallback baseline

### Tasks

- Select first Bayesian method
- Implement training or posterior fitting
- Generate predictive probabilities
- Generate uncertainty scores
- Compare against deterministic baseline
- Evaluate calibration and selective referral performance

### Deliverables

- Bayesian last-layer implementation
- Bayesian evaluation results
- Uncertainty metrics
- Calibration comparison
- Experiment log entry

## Phase 7 — Selective Referral Evaluation

### Goal

Evaluate whether uncertainty estimates can support safer referral or abstention behavior.

### Tasks

- Rank predictions by uncertainty
- Evaluate coverage versus accuracy
- Evaluate selective risk
- Test referral thresholds
- Compare deterministic confidence against Bayesian uncertainty
- Visualize referral trade-offs

### Deliverables

- Selective referral evaluation script
- Coverage-risk curves
- Referral threshold analysis
- Technical report section

## Phase 8 — Dataset Shift and Robustness

### Goal

Evaluate how models behave when the test distribution changes.

### Possible Shift Settings

- Cross-dataset testing
- Image quality degradation
- Subgroup or camera/source shift, if metadata is available
- Train/test distribution differences within the dataset

### Tasks

- Define shift scenario
- Evaluate deterministic baseline under shift
- Evaluate Bayesian last-layer model under shift
- Compare performance degradation
- Compare calibration degradation
- Compare uncertainty behavior

### Deliverables

- Shift evaluation results
- Robustness comparison
- Technical report section

## Phase 9 — Optional Fine-Tuning Extension

### Goal

Use A100/H100 cluster access to explore whether partial or full fine-tuning changes results.

### Tasks

- Define fine-tuning configuration
- Run limited fine-tuning experiment
- Compare frozen RETFound versus fine-tuned RETFound
- Re-run calibration and uncertainty evaluation
- Document compute cost and limitations

### Deliverables

- Fine-tuning experiment results, if completed
- Updated comparison table
- Limitations discussion

## Phase 10 — Final Report and Repository Polish

### Goal

Prepare the project as a research-grade GitHub portfolio artifact.

### Tasks

- Complete `docs/technical_report.md`
- Update `README.md`
- Add result tables
- Add key figures
- Clean scripts and configuration files
- Confirm reproducibility instructions
- Review logs and documentation
- Add limitations and future work

### Deliverables

- Final technical report
- Updated README
- Clean reproducible repository
- Final experiment logs
- Final decision log

## Recommended Execution Order

The recommended order is:

1. Finish project documentation skeleton
2. Complete literature review
3. Select dataset
4. Integrate RETFound
5. Extract frozen embeddings
6. Train deterministic baseline
7. Implement Bayesian last-layer model
8. Evaluate calibration and uncertainty
9. Evaluate selective referral
10. Evaluate dataset shift
11. Add optional fine-tuning
12. Finalize technical report and README

## Immediate Next Tasks

The next concrete tasks are:

1. Draft initial `docs/project_plan.md`
2. Commit project plan
3. Start `research_process/literature_review.md`
4. Identify candidate datasets
5. Identify RETFound checkpoint and integration path

## Git Workflow

Each meaningful documentation or code change should be committed with a clear message.

Example commit messages:

```bash
git add docs/project_plan.md
git commit -m "Add initial project plan"