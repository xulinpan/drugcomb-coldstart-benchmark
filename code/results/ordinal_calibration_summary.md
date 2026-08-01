# Ordinal error structure and calibration (reviewer 2, comments 3 and 4)

Standard grouped split, test partition (n = 73,618). Produced by
`experiments_ordinal_calibration.py`. Lookup baselines use the same inverse-frequency class
weights as the trained models, so hard predictions are comparable.

## Ordinal error structure (comment 3)

Classes are ordinal (antagonistic < additive < synergistic). RPS is the ranked probability score
(lower is better); MAOD is mean absolute ordinal distance; a distance-2 error is a synergistic pair
called antagonistic or vice versa.

| Model | RPS | MAOD | Distance-2 errors | Share of errors at distance 2 |
|---|---:|---:|---:|---:|
| Global prior (reference) | 0.1032 | 1.0169 | 9,234 | 14.1% |
| Logistic regression | 0.1106 | 0.3606 | 3,671 | 16.1% |
| Per-(cell line, drug) prior | **0.0674** | 0.2900 | 795 | **3.9%** |
| LightGBM | 0.0702 | 0.2282 | 738 | 4.6% |

The manuscript's claim is supported quantitatively: only 4.6% of LightGBM's errors cross the full
ordinal span, against 14.1% for a prior-only predictor and 16.1% for the linear baseline, and mean
absolute ordinal distance is lowest for LightGBM (0.228). Note that the memorization baseline attains a
marginally better RPS than the full model (0.0674 vs 0.0702), consistent with the finding that most of
the predictive signal is conditional base rates.

## Calibration (comment 4)

Expected calibration error (ECE) uses 15 uniform bins; per-class values are one-vs-rest. Brier is the
multiclass score. Reliability curves for **both** models are now plotted (Figure: calibration).

| Model | Top-label ECE | Brier (multiclass) | ECE antagonistic | ECE synergistic |
|---|---:|---:|---:|---:|
| Global prior | 0.0009 | 0.3857 | 0.0014 | 0.0005 |
| Logistic regression | 0.1108 | 0.4488 | 0.1488 | 0.1501 |
| Per-(cell line, drug) prior | 0.0442 | **0.2534** | 0.0472 | 0.0572 |
| LightGBM | **0.0236** | 0.2905 | 0.0786 | 0.0855 |

The qualitative phrase "reasonably calibrated" is replaced by these figures. LightGBM's top-label ECE
(0.024) is roughly a fifth of the logistic baseline's (0.111) and its Brier score is markedly better,
so the description was directionally right but should be stated numerically. The rare classes remain
over-confident under class re-weighting (per-class ECE 0.079 and 0.086 against 0.024 overall), which
quantifies the miscalibration the manuscript previously described only in words. The global prior is
trivially well calibrated but useless (Brier 0.386, MAOD 1.017), illustrating why calibration must be
read alongside discrimination.

## Grouped permutation importance (reviewer 2, minor comment 3)

Permutation importance is robust to feature *cardinality* relative to impurity/split-gain importance,
but it is not robust to feature *correlation*: permuting one member of a correlated group leaves the
other members intact as proxies, so the group's contribution is masked.

That is exactly the situation here. Every cell line in the dataset maps to exactly one tissue (100%),
so tissue is a deterministic function of cell-line identity. Permuting each in turn on a 20,000-row
test subsample (macro-F1 drop, LightGBM):

| Permuted | Macro-F1 drop |
|---|---:|
| Cell line alone | +0.2472 |
| Tissue alone | −0.0039 |
| Cell line and tissue jointly | +0.2558 |
| Sum of individual drops | +0.2434 |
| Joint − sum (super-additivity) | +0.0124 |

Tissue's near-zero individual importance therefore reflects masking by its perfect predictor rather
than genuine irrelevance, and the joint permutation exceeds the sum of the individual drops. The
manuscript's conclusion that tissue is redundant given cell line is sustained, but the supporting
argument is now the deterministic mapping plus the joint permutation, not the single-feature score.
