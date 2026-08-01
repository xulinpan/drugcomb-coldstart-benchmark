# Memorization baselines (reviewer concern 5)

Standard grouped split, test partition (n = 73,618). All lookup baselines are fitted on the training
partition only, smoothed toward the global prior with pseudo-count 20, backing off to the coarser
level for unseen keys. Because the reported models are class-weighted, hard predictions from the
lookup baselines are taken after applying the same inverse-frequency weights, so the comparison is
like-for-like; macro-AUC is weight-independent. Produced by `experiments_baselines.py`.

| Model | Balanced acc. | Macro-F1 | Macro-AUC | Interactions? |
|---|---:|---:|---:|---|
| Global class prior | 0.333 | 0.065 | 0.500 | none (reference) |
| Per-cell-line prior | 0.598 | 0.516 | 0.799 | none |
| Per-drug-pair prior | 0.611 | 0.510 | 0.782 | none |
| Logistic regression (manuscript baseline) | 0.633 | 0.559 | 0.821 | additive |
| LightGBM, depth 1 (additive by construction) | 0.653 | 0.562 | 0.842 | additive |
| **Per-(cell line, drug) prior** | **0.778** | **0.647** | **0.895** | drug x cell |
| Per-(cell line, drug pair) prior | 0.775 | 0.647 | 0.894 | pair x cell |
| **LightGBM, full (manuscript model)** | **0.805** | **0.703** | **0.922** | unrestricted |

## Interpretation

Two conclusions follow, and they pull in different directions.

1. **The interaction is real.** Every additive model plateaus near 0.60-0.65 balanced accuracy: single
   entity priors (0.598, 0.611), the linear baseline (0.633) and a depth-1 LightGBM that cannot
   represent any feature interaction (0.653). Performance only rises once drug identity and cell-line
   identity are combined. The manuscript's claim that a linear model cannot capture the relevant
   structure is therefore supported.

2. **But the interaction is memorization, not learned pharmacology.** A smoothed per-(cell line, drug)
   base-rate lookup table - which does no learning beyond counting training labels - reaches 0.778
   balanced accuracy and 0.895 macro-AUC. The full gradient-boosted model adds only 0.027 balanced
   accuracy and 0.027 macro-AUC beyond that table.

The reviewer's concern is therefore justified in substance: most of the headline performance is
recoverable from conditional base rates. This is consistent with, and strengthens, the paper's central
argument - the model succeeds on known entities by memorizing their empirical synergy rates, which is
exactly why performance collapses when those entities are unseen (leave-cell-line-out 0.483,
leave-study-out 0.325).

The manuscript's Discussion has been revised accordingly: the LightGBM advantage is attributed to
conditional (drug x cell) base rates captured as an interaction, not to learned pharmacological
mechanism.
