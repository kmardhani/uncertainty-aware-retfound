
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


## DDR Evaluation Of APTOS-Trained SNGP-Style Head

The APTOS-trained SNGP-style cached-feature head was evaluated on DDR cached RETFound features as a second-dataset validation test. The model checkpoint selected by APTOS validation sensitivity was evaluated directly on DDR without retraining or refitting the SNGP diagonal precision state.

### Default-Threshold Result

At threshold `0.50`, DDR performance was poor:

1. Accuracy: `0.5990`
2. Sensitivity: `0.2701`
3. Specificity: `0.8675`
4. Balanced accuracy: `0.5688`
5. AUC: `0.5833`
6. ECE: `0.2168`
7. NLL: `0.9340`
8. Brier score: `0.2977`
9. Confusion matrix: `[[897, 137], [616, 228]]`

The main failure was a high false-negative count:

    false negatives = 616

This indicates poor transfer from APTOS to DDR for the SNGP sensitivity-selected checkpoint.

### Threshold Tuning

Threshold tuning did not solve the cross-dataset failure.

The best balanced-accuracy threshold was `0.45`, with:

1. Sensitivity: `0.3069`
2. Specificity: `0.8424`
3. Balanced accuracy: `0.5746`
4. False negatives: `585`
5. False positives: `163`

A very low threshold of `0.01` reduced false negatives to `10`, but produced `1018` false positives and specificity of only `0.0155`. This is not a practical operating point.

### Selective Referral

Selective referral at approximately `80%` coverage also failed to recover a useful screening behavior on DDR.

At approximately `80%` coverage:

1. `predictive_entropy`: false negatives `519`, false positives `59`, sensitivity `0.2160`
2. `sngp_variance`: false negatives `478`, false positives `133`, sensitivity `0.3042`
3. `sngp_uncertainty`: false negatives `521`, false positives `60`, sensitivity `0.2165`

Referral mainly removed false positives rather than false negatives, which is not the desired safety behavior for referable-DR screening.

### Interpretation

The SNGP-style cached-feature head should be treated as a useful internal-validation baseline but not as a successful cross-dataset robustness method in its current form.

The APTOS result showed that SNGP-style training could produce a high-sensitivity source-dataset operating point. However, DDR validation showed weak discrimination, poor calibration, and poor false-negative control.

This negative result is scientifically useful because it shows that distance-aware uncertainty approximations do not automatically solve dataset shift in frozen RETFound feature space. The stronger project-level conclusion is that uncertainty-aware methods must be evaluated under external dataset shift, and that threshold tuning or selective referral can fail when the underlying cross-dataset ranking is weak.


## Consolidated DDR Results

A consolidated DDR comparison was constructed across native DDR softmax, native DDR variational Bayesian heads, threshold sweeps, selective referral, and the APTOS-trained SNGP-style head evaluated on DDR.

### Full-Coverage DDR Models

The native DDR softmax model achieved balanced full-coverage performance:

1. Sensitivity: `0.7773`
2. Specificity: `0.8075`
3. Balanced accuracy: `0.7924`
4. AUC: `0.8688`
5. False negatives: `188`
6. False positives: `199`

The native DDR Bayesian model selected by validation loss achieved similar balanced accuracy with higher specificity but lower sensitivity:

1. Sensitivity: `0.7275`
2. Specificity: `0.8530`
3. Balanced accuracy: `0.7902`
4. AUC: `0.8782`
5. False negatives: `230`
6. False positives: `152`

The native DDR Bayesian model selected by sensitivity produced the strongest safety-oriented full-coverage operating point:

1. Sensitivity: `0.8472`
2. Specificity: `0.7031`
3. Balanced accuracy: `0.7751`
4. AUC: `0.8634`
5. False negatives: `129`
6. False positives: `307`

In contrast, the APTOS-trained SNGP-style model transferred poorly to DDR:

1. Sensitivity: `0.2701`
2. Specificity: `0.8675`
3. Balanced accuracy: `0.5688`
4. AUC: `0.5833`
5. False negatives: `616`
6. False positives: `137`

### Threshold Tuning

Threshold tuning showed that false negatives can be reduced by lowering the positive-class threshold, but this often creates a large false-positive burden. For example, the DDR Bayesian sensitivity-selected model reached zero false negatives at threshold `0.03`, but produced `979` false positives. The same pattern appeared for softmax and Bayesian val-loss-selected models.

This demonstrates that false-negative reduction is not unique to Bayesian modeling. The clinically meaningful question is the full tradeoff among sensitivity, specificity, false positives, false negatives, and referral burden.

### Selective Referral

At approximately `80%` accepted coverage, selective referral improved the DDR Bayesian sensitivity-selected model.

Using confidence or predictive entropy as the uncertainty signal, the Bayesian model achieved:

1. Coverage: `0.8003`
2. Referral rate: `0.1997`
3. Sensitivity: `0.8966`
4. Specificity: `0.7596`
5. Balanced accuracy: `0.8281`
6. False negatives: `72`
7. False positives: `194`

This reduced accepted-case false negatives from `129` at full coverage to `72` at approximately `80%` coverage.

The SNGP-style model did not show useful selective-referral behavior under DDR shift. At approximately `80%` coverage, SNGP uncertainty signals still left hundreds of accepted-case false negatives, with sensitivity between approximately `0.216` and `0.304`.

### Interpretation

The consolidated DDR results support three conclusions.

First, Bayesian sensitivity selection provides the strongest full-coverage safety-oriented DDR operating point among the native DDR models tested.

Second, selective referral can improve accepted-case reliability when the model has a useful uncertainty-ranking signal. For the DDR Bayesian sensitivity-selected model, referral at approximately `20%` reduced accepted-case false negatives substantially while improving balanced accuracy.

Third, the SNGP-style cached-feature head did not improve cross-dataset robustness in this implementation. Its strong APTOS internal validation result did not transfer to DDR, and its uncertainty measures did not reliably identify dangerous missed-positive cases.

These findings reinforce the project's central methodological point: uncertainty-aware methods must be evaluated under external dataset shift, and internal validation performance alone is not sufficient evidence of clinical screening robustness.

