# A Leakage-Controlled Cold-Start Benchmark for Drug-Combination Synergy Classification

Reproducible code and results for the manuscript *A leakage-controlled cold-start benchmark for
drug-combination synergy classification* (BMC Bioinformatics, under review).

The study asks how much drug-combination synergy is predictable from **pre-treatment identity and
context alone** — the two compound identities, cell line, tissue, study, clinical-development phase
and catalogued protein targets — with every response-derived variable excluded from the features. It
is deliberately a *lower-bound* benchmark, not a competitor to molecular or omics models, and its main
contribution is the evaluation framework: leakage control plus cold-start regimes in which the
entities themselves are unseen.

## Headline results

Standard grouped split, DrugComb v1.5 (739,964 combinations; 26 studies, 17 tissues, 288 cell lines,
4,268 compounds). Three-class ZIP endpoint at ±10.

| Model | Balanced acc. | Macro-F1 | Macro-AUC |
|---|---:|---:|---:|
| Majority baseline | 0.333 | 0.289 | – |
| Logistic regression | 0.633 | 0.559 | 0.821 |
| Per-(cell line, drug) base-rate lookup | 0.778 | 0.647 | 0.895 |
| **LightGBM** | **0.805** | **0.703** | **0.922** |

Performance under distribution shift — this is the point of the benchmark:

| Regime | Balanced acc. |
|---|---:|
| Standard (novel pairings of known entities) | 0.805 |
| Leave-drug-out | 0.691 ± 0.004 |
| Leave-cell-line-out | 0.483 ± 0.010 |
| Leave-tissue-out | 0.426 ± 0.007 |
| Leave-study-out | 0.325 ± 0.009 (random floor = 0.333) |

A smoothed base-rate lookup table reaches 0.778 of the model's 0.805, so most of the standard-split
performance is conditional memorization of entity-specific synergy rates rather than learned
pharmacology — which is precisely why it collapses when entities are unseen.

## Data

The benchmark is built from the public DrugComb v1.5 summary release:

- DrugComb portal: <https://drugcomb.org/download/>
- Zenodo record: <https://zenodo.org/records/15235991>

`src/prep.py` derives the modeling table (`drugcomb_synergy_prediction_modeling.csv`) from the summary
file. No restricted or patient-identifiable data are used. The modeling table is ~900 MB and is not
included in this repository.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python ≥ 3.10. The optional molecular baseline additionally needs `rdkit`.

## Usage

All scripts auto-discover `drugcomb_synergy_prediction_modeling.csv` in the working directory or its
parents; otherwise set the data path explicitly:

```bash
export DRUGCOMB_DATA=/path/to/drugcomb_synergy_prediction_modeling.csv
export DRUGCOMB_OUT=outputs          # where intermediate artefacts are written
```

### Primary pipeline

```bash
python src/prep.py                                   # build the cached feature table
python src/train_lr.py                               # logistic baseline
python src/train_gbm.py                              # LightGBM
python src/analysis.py boot pr perm tissue colddrug coldcell
python src/figures.py && python src/figures2.py
```

### Review analyses

Each script is independent, resumable, and writes JSON to `results/`.

```bash
# study confounding and cross-context transfer
python src/experiments_review.py all

# drug-pair symmetry / reverse-pair contamination audit
python src/experiments_symmetry.py all

# multi-seed cold-start stability + leave-drug-out S1/S2 decomposition
python src/experiments_multiseed.py --regime colddrug   --seeds 42 43 44 45 46
python src/experiments_multiseed.py --regime coldcell   --seeds 42 43 44 45 46
python src/experiments_multiseed.py --regime coldtissue --seeds 42 43 44 45 46
python src/experiments_multiseed.py --regime coldstudy  --seeds 42 43 44 45 46
python src/experiments_multiseed.py --summary

# memorization baselines (what the model adds over base rates)
python src/experiments_baselines.py all

# classification-threshold sensitivity and cluster-bootstrap intervals
python src/experiments_threshold_bootstrap.py threshold nearcut bootstrap

# ordinal error structure (RPS) and calibration (ECE / Brier), both models
python src/experiments_ordinal_calibration.py lr
python src/experiments_ordinal_calibration.py ordinal calibration

# figures at 300 DPI
python src/regen_figures_300dpi.py
```

### Optional: molecular comparator

Requires a drug→SMILES table and RDKit. Provided as a scaffold for future comparison; it was **not**
used for the reported benchmark results.

```bash
pip install rdkit
python src/fetch_drug_smiles.py --drugcomb drugcomb_drugs.csv --out drug_smiles.csv
python src/fingerprint_baseline.py --smiles drug_smiles.csv --mode struct_ctx --regime standard
```

`src/drug_smiles_demo.csv` contains a small verified example table (14 compounds, PubChem SMILES)
sufficient to smoke-test the pipeline.

## Repository layout

```
src/        analysis and experiment scripts
results/    machine-readable metrics (JSON) and per-analysis summaries (Markdown)
figures/    manuscript figures, 300 DPI
```

`results/*_summary.md` files describe each analysis in prose with its result tables:
`review_experiments_summary.md`, `multiseed_summary.md`, `baselines_summary.md`,
`threshold_bootstrap_summary.md`, `ordinal_calibration_summary.md`, `menden_comparison.md`.

## Reproducibility notes

- A fixed seed (42) is used throughout; `seed=42` reproduces every value reported in the manuscript.
- Cold-start regimes vary only the entity holdout; the model's own random state stays fixed at 42.
- Runtime is roughly 30–40 s per LightGBM fit on 2 CPU cores; the full five-seed multi-seed protocol
  takes about 20 minutes.
- The reported multi-seed figures in the manuscript were obtained over 2–3 draws per regime; the
  scripts run the full five-seed protocol.

## Citation

See `CITATION.cff`. Please cite the manuscript and, if you use the underlying data, the DrugComb
portal papers (Zagidullin et al. 2019; Zheng et al. 2021) and the ZIP score (Yadav et al. 2015).

## License

MIT — see `LICENSE`. The DrugComb data are distributed under their own terms; see the DrugComb portal.
