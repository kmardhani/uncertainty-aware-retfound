
## SNGP-Style Cached-Feature Head

The project now includes an SNGP-style distance-aware cached-feature head for frozen RETFound embeddings. This method is intentionally limited to cached RETFound feature vectors and should not be described as full end-to-end SNGP RETFound.

### Method

The SNGP-style head uses:

1. Frozen RETFound cached features as input
2. Optional spectral-normalized projection
3. Fixed random Fourier features
4. Linear classifier
5. Diagonal precision estimate over random-feature activations

The diagonal variance proxy is computed as:

    sngp_variance(x) = sum_j phi_j(x)^2 / precision_diag_j

where `phi(x)` is the random Fourier feature representation.

The model also reports predictive entropy:

    predictive_entropy = -sum_c p_c log p_c

The current combined uncertainty score is:

    sngp_uncertainty = predictive_entropy + sngp_variance

This combined score is unnormalized and uncalibrated. It should be interpreted as an exploratory score rather than a validated calibrated uncertainty estimate.

### APTOS Internal Validation

The sensitivity-selected SNGP-style head achieved:

1. Accuracy: `0.9098`
2. Sensitivity: `0.9805`
3. Specificity: `0.8585`
4. Balanced accuracy: `0.9195`
5. AUC: `0.9711`
6. ECE: `0.0287`
7. Confusion matrix: `[[182, 30], [3, 151]]`

This gives a strong screening-oriented operating point on APTOS validation.

Threshold analysis showed that false negatives can be further reduced by lowering the decision threshold. At threshold `0.40`, the model had `1` false negative and `37` false positives. At threshold `0.20`, the model reached `0` false negatives but increased false positives to `67`.

### Selective Referral

For the SNGP sensitivity-selected checkpoint, selective referral was evaluated with:

1. `predictive_entropy`
2. `sngp_variance`
3. `sngp_uncertainty`

At approximately `80%` coverage, `predictive_entropy` produced the strongest accepted-case result:

1. Accepted cases: `293 / 366`
2. Deferred cases: `73 / 366`
3. Referral rate: `0.1995`
4. Accepted-case false negatives: `0`
5. Accepted-case false positives: `14`
6. Accepted-case sensitivity: `1.0000`
7. Accepted-case specificity: `0.9146`
8. Accepted-case accuracy: `0.9522`

The raw combined `sngp_uncertainty` score was slightly weaker, and `sngp_variance` alone was substantially weaker.

### Interpretation

The current SNGP-style head is useful as an additional uncertainty-aware cached-feature baseline. However, APTOS diagnostics show that its diagonal variance proxy is not a strong standalone referral signal on internal validation.

The main methodological implication is that uncertainty-aware selective referral remains valuable, but the strongest APTOS referral signal is predictive entropy rather than the current diagonal SNGP variance estimate.

The SNGP-style variance proxy should still be evaluated on DDR, because its intended value is distance-aware behavior under dataset shift. The central question for the next stage is whether `sngp_variance` becomes more useful under APTOS-to-DDR transfer.

