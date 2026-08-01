"""Stage 1: load CSV, engineer features, cache to parquet for fast re-loading."""
import os
import time
import numpy as np, pandas as pd
from pathlib import Path

DATA = os.environ.get("DRUGCOMB_DATA", "drugcomb_synergy_prediction_modeling.csv")
OUT = Path(os.environ.get("DRUGCOMB_OUT", "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["antagonistic", "additive", "synergistic"]
C2I = {c: i for i, c in enumerate(CLASSES)}
CAT_COLS = ["study_name", "tissue_name", "cell_line_name", "drug_row", "drug_col"]
NUM_PASSTHROUGH = ["drug_row_clinical_phase", "drug_col_clinical_phase"]

t0 = time.time()
use = CAT_COLS + NUM_PASSTHROUGH + ["drug_row_target_name", "drug_col_target_name",
                                    "zip_synergy_label", "split"]
df = pd.read_csv(DATA, usecols=use)
df["y"] = df["zip_synergy_label"].map(C2I).astype(np.int8)

def n_targets(s):
    if not isinstance(s, str) or s == "":
        return 0
    return s.count(";") + 1

df["n_targets_row"] = df["drug_row_target_name"].map(n_targets).astype(np.int16)
df["n_targets_col"] = df["drug_col_target_name"].map(n_targets).astype(np.int16)
df["has_target_row"] = (df["n_targets_row"] > 0).astype(np.int8)
df["has_target_col"] = (df["n_targets_col"] > 0).astype(np.int8)

# vectorized shared-target: split into sets
def to_set(s):
    if not isinstance(s, str) or s == "":
        return frozenset()
    return frozenset(t.strip() for t in s.split(";"))
sr = df["drug_row_target_name"].map(to_set)
sc = df["drug_col_target_name"].map(to_set)
df["shared_target"] = [int(len(a & b) > 0) for a, b in zip(sr, sc)]
df["shared_target"] = df["shared_target"].astype(np.int8)

for c in NUM_PASSTHROUGH:
    df[c] = df[c].fillna(-1).astype(np.float32)

keep = CAT_COLS + NUM_PASSTHROUGH + ["n_targets_row", "n_targets_col",
        "has_target_row", "has_target_col", "shared_target", "y", "split"]
df[keep].to_parquet(OUT / "processed.parquet", index=False)
print(f"Saved processed.parquet  rows={len(df):,}  in {time.time()-t0:.1f}s")
