# Multi-seed cold-start evaluation (reviewer concern 4) and leave-drug-out decomposition (concern 3)

LightGBM, identical hyperparameters and class weighting to the primary analysis. Seed varies the
entity holdout only; the model's own random_state is fixed at 42. Produced by
`experiments_multiseed.py`. Seed 42 reproduces every published value exactly
(colddrug 0.694, coldcell 0.482, coldtissue 0.431, coldstudy 0.332).

## Concern 4 — stability across holdout draws

| Regime | Seeds | Balanced acc. mean ± SD [min, max] | Macro-F1 | Macro-AUC | n_test range |
|---|---:|---|---|---|---|
| Leave-drug-out | 2 | 0.691 ± 0.004 [0.688, 0.694] | 0.634 ± 0.004 | 0.859 ± 0.006 | 139,495–175,617 |
| Leave-cell-line-out | 3 | 0.483 ± 0.010 [0.474, 0.493] | 0.478 ± 0.013 | 0.798 ± 0.023 | 118,128–126,905 |
| Leave-tissue-out | 2 | 0.426 ± 0.007 [0.421, 0.431] | 0.461 ± 0.008 | 0.757 ± 0.021 | 111,231–133,269 |
| Leave-study-out | 2 | 0.325 ± 0.009 [0.319, 0.332] | 0.298 ± 0.019 | 0.453 ± 0.045 | 218,197–406,267 |

Balanced accuracy is stable across draws in every regime (SD ≤ 0.010), even though the test-set size
varies substantially — most strikingly for leave-study-out, where n_test ranges from 218,197 to
406,267 (a 1.9-fold difference) while balanced accuracy moves only from 0.319 to 0.332. Macro-AUC is
the least stable statistic, particularly for leave-study-out (0.421–0.486).

The published single-seed numbers are therefore representative rather than fortunate draws, and the
ordering of the regimes (drug > cell line > tissue > study) is preserved in every draw.

## Concern 3 — leave-drug-out decomposed

| Seed | Subset | n | Balanced acc. | Macro-F1 | Macro-AUC |
|---|---|---:|---:|---:|---:|
| 42 | Pooled (as published) | 139,495 | 0.6940 | 0.6310 | 0.8547 |
| 42 | S1: exactly one unseen drug | 132,630 | 0.6982 | 0.6355 | 0.8580 |
| 42 | S2: both drugs unseen | 6,865 | 0.6110 | 0.5499 | 0.7857 |
| 43 | Pooled | 175,617 | 0.6882 | 0.6366 | 0.8625 |
| 43 | S1: exactly one unseen drug | 163,428 | 0.6930 | 0.6408 | 0.8670 |
| 43 | S2: both drugs unseen | 12,189 | 0.5924 | 0.5556 | 0.7907 |

The reviewer is correct that the pooled figure is dominated by S1: rows with both drugs unseen are only
4.9% (seed 42) and 6.9% (seed 43) of the cold-drug test set. S2 is consistently harder than S1 by
about 0.09–0.10 balanced accuracy and about 0.07 macro-AUC. S2 nevertheless remains well above the
0.333 random floor, so prediction for entirely novel compound pairs degrades substantially but does
not collapse.

## Note

Seeds obtained under the 45-second execution limit of the analysis environment: 2–3 per regime.
`experiments_multiseed.py --regime <r> --seeds 42 43 44 45 46` reproduces and extends this to the
full five-seed protocol (~35 s per run); one leave-study-out draw (seed 44) exceeded the limit here
and is not included.
