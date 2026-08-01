# Threshold sensitivity (concern 7) and dependence-aware intervals (concern 6)

Standard grouped split, LightGBM, seed 42. Produced by `experiments_threshold_bootstrap.py`.

## Threshold sensitivity

Re-deriving the endpoint at other |ZIP| cut-offs and retraining:

| Cut-off | Class share (antag./add./syn.) | Balanced acc. | Macro-F1 | Macro-AUC |
|---|---|---:|---:|---:|
| ±5 | 0.246 / 0.533 / 0.222 | 0.727 | 0.702 | 0.872 |
| **±10 (primary)** | 0.107 / 0.767 / 0.126 | **0.805** | **0.703** | **0.922** |
| ±15 | 0.050 / 0.868 / 0.082 | 0.853 | 0.709 | 0.951 |
| ±20 | 0.026 / 0.917 / 0.056 | 0.861 | 0.706 | 0.963 |

Macro-F1 is essentially invariant across the range (0.702–0.709), so the substantive conclusions do
not depend on the ±10 convention. Balanced accuracy and macro-AUC increase monotonically with a
stricter cut-off, which is expected: larger |ZIP| values are more extreme and more separable, while the
additive class grows from 53% to 92% of the data. The ±10 convention is therefore neither especially
favourable nor especially unfavourable to the model.

## Near-threshold exclusion

Using the primary (±10) model and dropping test rows whose ZIP score lies within a margin of the cut:

| Margin | Rows excluded | Balanced acc. | Macro-F1 |
|---|---:|---:|---:|
| none | 0 | 0.805 | 0.703 |
| ±1 | 4,352 (5.9%) | 0.831 | 0.717 |
| ±2 | 8,820 (12.0%) | 0.851 | 0.729 |
| ±5 | 24,433 (33.2%) | 0.888 | 0.768 |

Performance improves steadily as boundary-adjacent pairs are removed, confirming that a material part
of the residual error is label ambiguity at the cut rather than model failure.

## Dependence-aware confidence intervals

Rows are not independent: each row is a distinct assay block (block_id is unique per row in the
modeling table, so there are no within-block replicates), but rows share cell lines and compounds. The
95% interval for balanced accuracy on the standard split:

| Resampling unit | 95% CI | Width | Clusters |
|---|---|---:|---:|
| Rows (i.i.d., as in the primary analysis) | [0.8012, 0.8091] | 0.0079 | – |
| Whole compounds (drug_row) | [0.7979, 0.8115] | 0.0136 | 1,633 |
| Whole cell lines | [0.7870, 0.8135] | 0.0265 | 276 |

The reviewer is correct that i.i.d. row resampling understates uncertainty: clustering by cell line
widens the interval by a factor of about 3.4. The revised manuscript reports the cell-line cluster
bootstrap as the primary interval. The comparison with the logistic baseline (0.633) is unaffected,
since the gap greatly exceeds even the widened interval.
