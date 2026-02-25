# A/B Testing Methodology

This document details the experimental design, hypothesis, metrics, statistical tests, and success criteria for the A/B testing framework.

## Objective

Compare two machine learning model variants serving the same prediction task to determine which model delivers superior performance in a production-like environment. The framework enables data-driven model selection by measuring both technical and operational metrics under controlled conditions.

## Hypothesis

**Null Hypothesis (H₀):** There is no statistically significant difference in performance between Model A (Logistic Regression) and Model B (XGBoost) across the selected metrics.

**Alternative Hypothesis (H₁):** There exists a statistically significant difference in at least one key metric between the two model variants.

### Model Descriptions

| Property | Model A | Model B |
|----------|---------|---------|
| Algorithm | Logistic Regression | XGBoost |
| Preprocessing | StandardScaler pipeline | Raw features |
| Training Data | Breast Cancer Wisconsin (455 samples) | Breast Cancer Wisconsin (455 samples) |
| Test Data | 114 held-out samples | 114 held-out samples |

## Experimental Design

### Traffic Assignment

- **Default split:** 50/50 random assignment
- **Randomization unit:** Individual request (stateless) or user session (sticky)
- **Sticky assignment:** Deterministic SHA-256 hash of session ID ensures the same user always sees the same variant
- **Configurable at runtime:** The split ratio can be adjusted via API without redeployment

### Sample Size Considerations

For a two-proportion z-test with:
- Baseline conversion rate: ~50% (balanced classes)
- Minimum detectable effect: 5 percentage points
- Significance level (α): 0.05
- Statistical power (1 − β): 0.80

The required sample size per group is approximately **385 observations**. A minimum of 500 total requests (250 per variant at 50/50 split) is recommended.

### Data Collection

For every prediction request, the system logs:

| Field | Type | Purpose |
|-------|------|---------|
| `request_id` | UUID | Unique request identifier |
| `timestamp` | ISO 8601 | Time of prediction |
| `model_variant` | String (A/B) | Which model served the request |
| `input_features` | JSON array | Input feature vector |
| `prediction` | Float | Model's predicted class (0 or 1) |
| `prediction_probability` | Float | Confidence score |
| `latency_ms` | Float | End-to-end prediction time |

## Metrics

### Technical Metrics

1. **Positive Prediction Rate:** Proportion of positive predictions (class 1) per variant. Acts as a proxy for model behavior consistency.
2. **Mean Prediction Probability:** Average confidence score — higher values indicate more decisive predictions.
3. **Prediction Distribution:** The overall shape of prediction outputs, tested via non-parametric methods.

### Operational Metrics

1. **Mean Latency (ms):** Average prediction serving time — critical for user experience.
2. **P95 Latency (ms):** 95th percentile latency — captures tail latency affecting worst-case users.
3. **P99 Latency (ms):** 99th percentile latency — identifies extreme outliers.
4. **Error Rate:** Proportion of failed predictions — any increase indicates model instability.

### Business-Proxy Metrics

1. **Conversion Rate (Positive Prediction Rate):** For classification tasks, the positive prediction rate serves as a conversion proxy. A model that predicts positive more often may indicate higher engagement potential.
2. **Traffic Share:** Actual percentage of traffic served by each variant — validates that the split mechanism is functioning correctly.

## Statistical Tests

### 1. Welch's t-test (Latency)

- **Metric:** `latency_ms` (continuous)
- **Purpose:** Determine if there's a significant difference in mean prediction latency between variants
- **Assumptions:** Independent samples, approximately normal distribution (satisfied by CLT for n > 30)
- **Does not assume equal variances** (Welch's correction applied)

### 2. Mann-Whitney U Test (Predictions)

- **Metric:** `prediction` (ordinal/continuous)
- **Purpose:** Non-parametric test for differences in prediction distributions
- **Chosen because:** Predictions may not be normally distributed; this test is robust to non-normality
- **Alternative:** Two-sided (tests for any difference, not directional)

### 3. Chi-Squared Test (Prediction Categories)

- **Metric:** Contingency table of `model_variant` × `prediction` class
- **Purpose:** Test for association between model variant and prediction outcome
- **Appropriate for:** Categorical outcomes (binary classification: 0 vs 1)
- **Requirement:** Expected cell counts ≥ 5 (validated automatically)

### 4. Welch's t-test (Prediction Probability)

- **Metric:** `prediction_probability` (continuous, 0-1 range)
- **Purpose:** Determine if model confidence levels differ significantly
- **Interpretation:** Higher mean probability suggests more decisive predictions

## Significance Level

- **α = 0.05** (5% significance level)
- Results with p-value < 0.05 are considered statistically significant
- No multiple comparison correction is applied as each test addresses a different hypothesis

## Success Criteria for Selecting a Winning Model

A model is declared the winner if it satisfies **all** of the following:

1. **Statistical Significance:** At least one primary metric shows a statistically significant improvement (p < 0.05)
2. **Latency Parity:** Mean latency must not be significantly worse (>20% increase)
3. **No Degradation:** No statistically significant regression in any monitored metric
4. **Sufficient Sample Size:** Minimum 250 observations per variant

If no model meets all criteria, the recommendation is to:
- Extend the experiment duration
- Increase traffic volume
- Reassess the minimum detectable effect size

## Limitations

1. **No real-time ground truth:** This framework compares model behavior (predictions, latency) rather than actual correctness against labels. In production, a feedback loop with ground truth labels would enable accuracy/AUC comparison.
2. **Sequential testing bias:** The current analysis runs post-hoc. Continuous monitoring with sequential testing (e.g., using a spending function) would reduce false positive risk for ongoing experiments.
3. **Feature drift:** The framework does not currently detect input feature drift, which could invalidate results if the data distribution changes during the experiment.

## Reproducibility

All experiments are reproducible:
- Models trained with `random_state=42`
- Traffic simulation uses configurable seeds
- Database logs provide a complete audit trail
- Analysis pipeline reads directly from the persistent database
