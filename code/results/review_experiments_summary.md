# Review experiments — summary (LightGBM, seed 42)

| Regime | Bal. acc. | Macro-F1 | Macro-AUC | Rec. Antag. | Rec. Add. | Rec. Syn. |
|---|---|---|---|---|---|---|
| Standard, with study | 0.807 | 0.708 | 0.923 | 0.840 | 0.778 | 0.802 |
| Standard, no study | 0.806 | 0.708 | 0.922 | 0.838 | 0.778 | 0.803 |
| Leave-study-out (14/26 studies, n_test=406267) | 0.332 | 0.312 | 0.421 | 0.000 | 0.985 | 0.010 |
| Leave-tissue-out (7/17 tissues, n_test=111231) | 0.431 | 0.466 | 0.772 | 0.149 | 0.976 | 0.169 |
