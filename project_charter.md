# Project Charter

## Project Title

**Uncertainty-Aware RETFound for Trustworthy Retinal Disease Classification**

## Project Summary

This project investigates whether Bayesian last-layer adaptation can improve the trustworthiness of retinal disease classification models built on top of RETFound, a retinal foundation model.

The core idea is to use RETFound as a fixed or lightly adapted feature extractor and compare a standard deterministic classification head against uncertainty-aware alternatives, especially Bayesian last-layer methods. The project will focus not only on classification performance, but also on calibration, uncertainty estimation, selective referral, and robustness under dataset shift.

The intended outcome is a research-grade, reproducible GitHub project that demonstrates a principled approach to uncertainty-aware medical imaging classification.

## Motivation

Deep learning models for retinal disease classification can achieve strong predictive performance, but high accuracy alone is not sufficient for clinical or safety-sensitive settings. In real-world screening workflows, a model should also be able to communicate when it is uncertain, especially when encountering ambiguous images, poor-quality inputs, rare disease patterns, or data from a different distribution than the training set.

RETFound provides a strong foundation model for retinal imaging. However, using a powerful pretrained model does not automatically guarantee well-calibrated probabilities or reliable uncertainty estimates. This project explores whether Bayesian adaptation at the final classification layer can improve the reliability of predictions without requiring full retraining of the foundation model.

## Research Question

**Does Bayesian last-layer adaptation on top of RETFound improve calibration, uncertainty estimation, selective referral, and robustness to dataset shift compared with a standard deterministic classification head?**

## Objectives

The project has four main objectives:

1. **Build a reproducible RETFound-based retinal disease classification pipeline**
   - Load retinal image datasets.
   - Apply consistent preprocessing and train/validation/test splits.
   - Extract features using RETFound or fine-tune selected components if feasible.
   - Train a baseline deterministic classifier.

2. **Implement uncertainty-aware last-layer adaptation**
   - Use RETFound representations as input features.
   - Implement one or more Bayesian or approximate Bayesian classification heads.
   - Compare against standard softmax-based classification.

3. **Evaluate trustworthiness beyond accuracy**
   - Measure classification performance.
   - Measure calibration quality.
   - Evaluate uncertainty estimates.
   - Study selective prediction or referral behavior.
   - Test robustness under dataset shift where possible.

4. **Produce a research-grade public artifact**
   - Maintain clear experiment logs.
   - Document design decisions.
   - Track AI-assisted development transparently.
   - Write a technical report summarizing methods, experiments, results, and limitations.

## Scope

### In Scope

This project includes:

- RETFound-based retinal image classification.
- Baseline deterministic classifier.
- Bayesian or approximate Bayesian last-layer classifier.
- Calibration evaluation.
- Uncertainty estimation evaluation.
- Selective referral experiments.
- Dataset-shift or cross-dataset robustness experiments, if suitable datasets are available.
- Reproducible experiment tracking.
- Documentation suitable for a research-oriented GitHub portfolio project.

### Out of Scope

This project does not aim to:

- Develop a clinically deployable diagnostic system.
- Claim medical validity or regulatory readiness.
- Replace professional clinical judgment.
- Perform full-scale foundation model pretraining.
- Guarantee state-of-the-art performance across all retinal disease benchmarks.
- Use private clinical data unless appropriate permissions and safeguards are available.

## Methodology Overview

The project will follow a staged research workflow.

### Stage 1 — Environment and Repository Setup

- Create a reproducible Python environment using `uv`.
- Pin Python to version 3.11.
- Define project structure.
- Set up documentation and experiment logging.
- Confirm hardware constraints and compute strategy.

### Stage 2 — Literature Review

- Review RETFound and retinal foundation model literature.
- Review uncertainty estimation methods for deep learning.
- Review calibration metrics and selective prediction.
- Review uncertainty-aware medical imaging classification studies.
- Record relevant papers in `research_process/literature_review.md`.

### Stage 3 — Dataset Selection

Candidate datasets may include public retinal imaging datasets such as diabetic retinopathy or ocular disease classification datasets, depending on availability, licensing, and compatibility.

Dataset selection criteria:

- Publicly accessible or clearly permitted for research use.
- Suitable labels for classification.
- Sufficient sample size for baseline experiments.
- Potential for train/test or cross-dataset shift evaluation.
- Manageable compute requirements.

### Stage 4 — Baseline Model

The baseline model will use RETFound features with a standard deterministic classification head.

Possible baseline configuration:

- RETFound feature extractor.
- Linear or multilayer classification head.
- Cross-entropy loss.
- Softmax probabilities.
- Standard classification metrics.

### Stage 5 — Bayesian Last-Layer Adaptation

The uncertainty-aware model will keep the RETFound representation layer fixed or mostly fixed and replace the deterministic classification head with a Bayesian or approximate Bayesian alternative.

Candidate approaches may include:

- Bayesian logistic regression on RETFound embeddings.
- Laplace approximation for the final layer.
- Monte Carlo sampling from the posterior over final-layer weights.
- Ensemble-style approximation if needed as a practical fallback.

The exact method will be selected based on technical feasibility, available libraries, and compute constraints.

### Stage 6 — Evaluation

The project will compare deterministic and uncertainty-aware models using both predictive performance and trustworthiness metrics.

Classification metrics may include:

- Accuracy
- Balanced accuracy
- AUROC
- F1 score
- Sensitivity and specificity, where appropriate

Calibration metrics may include:

- Expected Calibration Error, or ECE
- Negative log-likelihood
- Brier score
- Reliability diagrams

Uncertainty and referral metrics may include:

- Predictive entropy
- Maximum softmax probability
- Mutual information, if supported by the Bayesian method
- Coverage versus accuracy curves
- Selective risk
- Referral rate at selected uncertainty thresholds

Dataset-shift evaluation may include:

- Cross-dataset testing
- Image quality perturbations
- Distribution shift between training and test subsets
- Performance and calibration degradation under shift

## Expected Deliverables

The project will produce the following deliverables:

1. **Working research codebase**
   - Data loading
   - Model training
   - Evaluation scripts
   - Experiment configuration

2. **Baseline experiment results**
   - Deterministic RETFound classifier
   - Standard performance metrics
   - Calibration metrics

3. **Uncertainty-aware experiment results**
   - Bayesian last-layer classifier
   - Uncertainty and calibration analysis
   - Selective referral results

4. **Dataset-shift analysis**
   - Robustness comparison between baseline and uncertainty-aware models
   - Discussion of limitations

5. **Technical report**
   - Problem statement
   - Related work
   - Methods
   - Experiments
   - Results
   - Limitations
   - Future work

6. **Research process documentation**
   - Literature review
   - Decision log
   - Experiment log
   - AI supervision log

## Success Criteria

The project will be considered successful if it demonstrates a clear, reproducible comparison between a deterministic RETFound-based classifier and an uncertainty-aware Bayesian last-layer variant.

A strong outcome would show that the Bayesian last-layer approach improves one or more of the following without causing unacceptable degradation in classification performance:

- Calibration quality
- Reliability of uncertainty estimates
- Selective referral behavior
- Robustness under dataset shift

A successful project does not require proving that the Bayesian method is universally superior. Negative or mixed results are acceptable if the experiments are rigorous, clearly documented, and honestly interpreted.

## Risks and Mitigations

### Risk 1 — RETFound integration may be difficult

RETFound code, checkpoints, or dependencies may require adaptation.

**Mitigation:** Start with a minimal feature extraction workflow. If needed, use frozen embeddings before attempting fine-tuning.

### Risk 2 — Compute limitations

Full fine-tuning may not be feasible on available hardware.

**Mitigation:** Prioritize frozen or partially frozen RETFound feature extraction and lightweight last-layer adaptation.

### Risk 3 — Dataset availability or licensing constraints

Some retinal datasets may have access restrictions.

**Mitigation:** Use only datasets with clear research permissions. Document dataset choices and limitations.

### Risk 4 — Bayesian methods may add complexity

Bayesian last-layer methods may be harder to implement and evaluate correctly.

**Mitigation:** Start with the simplest viable method, such as Bayesian logistic regression or Laplace approximation, before expanding to more complex approaches.

### Risk 5 — Results may not show clear improvement

The Bayesian last-layer approach may not outperform the deterministic baseline across all metrics.

**Mitigation:** Treat this as a valid research outcome. Focus on careful analysis, calibration behavior, selective referral trade-offs, and limitations.

## Ethical and Safety Considerations

This project is for research and educational purposes only. It is not intended for clinical deployment or medical decision-making.

The project will avoid making unsupported clinical claims. Any results will be framed as experimental findings based on selected datasets and evaluation protocols. Dataset documentation, limitations, and potential sources of bias will be clearly described.

Special care will be taken when discussing medical AI trustworthiness, including uncertainty, false positives, false negatives, dataset shift, and limitations of model predictions.

## Repository Documentation Plan

The repository will include the following key documentation files:

- `README.md`  
  Public-facing project overview, setup instructions, and high-level results.

- `project_charter.md`  
  Project purpose, scope, research question, objectives, risks, and success criteria.

- `docs/environment_plan.md`  
  Environment setup, dependency management, hardware notes, and reproducibility plan.

- `docs/project_plan.md`  
  Work breakdown, milestones, and implementation roadmap.

- `docs/technical_report.md`  
  Research-style report covering background, methods, experiments, results, and conclusions.

- `research_process/literature_review.md`  
  Notes on papers, methods, datasets, and related work.

- `research_process/decision_log.md`  
  Record of major design decisions and rationale.

- `research_process/experiment_log.md`  
  Chronological log of experiments, configurations, results, and observations.

- `research_process/ai_supervision_log.md`  
  Transparent record of AI-assisted planning, coding, writing, and review.

## Initial Milestones

### Milestone 1 — Project Foundation

- Complete initial repository structure.
- Draft project charter.
- Draft environment plan.
- Draft project plan.
- Identify candidate datasets.
- Identify RETFound integration strategy.

### Milestone 2 — Literature and Dataset Review

- Summarize key RETFound papers.
- Summarize uncertainty estimation and calibration methods.
- Select primary dataset.
- Define classification task.
- Document dataset limitations.

### Milestone 3 — Baseline Pipeline

- Implement data loading.
- Implement preprocessing.
- Extract RETFound features or integrate RETFound model.
- Train deterministic baseline classifier.
- Evaluate baseline performance and calibration.

### Milestone 4 — Bayesian Last-Layer Model

- Implement selected Bayesian last-layer method.
- Train uncertainty-aware classifier.
- Generate predictive uncertainty estimates.
- Compare against deterministic baseline.

### Milestone 5 — Selective Referral and Shift Evaluation

- Evaluate uncertainty-based referral thresholds.
- Analyze coverage versus accuracy.
- Test calibration and uncertainty under dataset shift.
- Compare robustness of both approaches.

### Milestone 6 — Final Report and Portfolio Polish

- Complete technical report.
- Update README with results and figures.
- Review reproducibility.
- Clean code and documentation.
- Prepare final GitHub project for public presentation.

## Working Assumptions

The project begins with the following assumptions:

- RETFound can be used as the retinal foundation model backbone or feature extractor.
- Public retinal image datasets can be used for experimentation.
- Frozen-feature or last-layer-only adaptation will be computationally feasible.
- Calibration and uncertainty metrics can be computed reliably.
- The project can be completed as an independent research-grade portfolio project without requiring clinical deployment.

These assumptions will be revisited as the project progresses.

## Definition of Done

The project is considered complete when:

- The repository can be set up from documented instructions.
- At least one deterministic RETFound-based baseline has been implemented and evaluated.
- At least one Bayesian or approximate Bayesian last-layer method has been implemented and evaluated.
- Calibration, uncertainty, and selective referral metrics have been reported.
- Dataset-shift behavior has been explored where feasible.
- The technical report clearly explains the methodology, results, limitations, and future work.
- The research process documentation is sufficiently complete to show how the project evolved.