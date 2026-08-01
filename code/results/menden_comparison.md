# Relation to the AstraZeneca–DREAM identity-only baseline (reviewer 2, comment 1)

## What Menden et al. (2019) actually report — verified against the primary source

*Community assessment to advance computational prediction of cancer drug combinations in a
pharmacogenomic screen*, Nature Communications 10:2674.

- The SC1 scoring metric was the **average weighted Pearson correlation** between predicted and
  observed continuous synergy.
- Following Team NAD's method, the organisers built a **baseline model using only cell-line and drug
  labels as input features**. It achieved a primary metric of **0.32** (Fig. 4a), described in the
  paper as "surprisingly high".
- For reference in the same paper: **experimental replicates** reached a primary metric of **0.43**
  (Fig. 3a), and **mean performance across all teams** — most using molecular data — was
  **r = 0.24 ± 0.01** (SC1A) and **0.23 ± 0.01** (SC1B).
- Adding molecular feature types to the NAD baseline improved performance significantly
  (t-test P = 0.009, 0.009, 0.002, 0.008), and drug target was the only feature that improved it when
  *swapped* for drug label (P = 0.012).

The identity-only baseline therefore reached roughly three-quarters of the experimental-replicate
ceiling and exceeded the average submitted model, while molecular features still added significantly.

## Our indicative counterpart

To place a number on the same scale we trained a continuous-ZIP regression variant of the same
identity-and-context model on the standard grouped split (LightGBM regressor, otherwise identical
settings):

| Statistic | Value |
|---|---:|
| Pearson r (all test rows) | 0.618 |
| Pearson r (\|ZIP\| ≤ 100; 99.9% of rows) | 0.610 |
| Spearman rho | 0.546 |
| Mean within-cell-line Pearson r (150 cell lines, n ≥ 30) | 0.476 |

**This is indicative, not like-for-like.** The datasets differ (DrugComb v1.5, 26 studies vs. the
AstraZeneca screen), the splits differ (our standard split places novel pairings of known entities in
the test set, whereas SC1B held out combinations), and the DREAM metric is a weighted correlation whose
exact weighting we do not reproduce. The within-cell-line mean (0.476) is the closest in spirit to a
stratified correlation and is the most conservative of our figures.

## How this is used in the manuscript

The agreement is qualitative and mutually reinforcing: an identity-only model performs far better than
one would expect from a predictor with no molecular information, in two independent datasets, under two
different endpoints. This external precedent complements the internal evidence in this paper — the
per-(cell line, drug) base-rate lookup reaching 0.778 balanced accuracy — and both point to the same
explanation, namely that much of the achievable signal is carried by entity-level base rates. Menden et
al. also show that molecular features add significantly on top of that baseline, which supports our
framing of the present benchmark as a lower bound rather than a replacement for molecular models.
