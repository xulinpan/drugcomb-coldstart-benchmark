"""
Drug-pair symmetry / reverse-pair leakage audit  (reviewer concern 1).

Answers three questions:
  1. AUDIT   How many test rows have a counterpart in training with drug A/B swapped?
             Reported for the standard split (and, with --regime, for cold-start splits).
  2. CLEAN   Do the reported metrics survive when every contaminated test row is removed?
  3. CANON   Do they survive an order-invariant (canonical) drug-pair encoding?

Usage:
  python experiments_symmetry.py audit
  python experiments_symmetry.py clean canon
  python experiments_symmetry.py all

Outputs review_experiments_output/symmetry_audit.json and prints a summary table.
Conventions match train_gbm.py / experiments_review.py (same features, class weights,
LightGBM hyperparameters, seed 42).
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             cohen_kappa_score, roc_auc_score, confusion_matrix,
                             classification_report)
import lightgbm as lgb

warnings.filterwarnings("ignore")
RNG = 42
CLASSES = ["antagonistic", "additive", "synergistic"]
C2I = {c: i for i, c in enumerate(CLASSES)}
CTX = ["study_name", "tissue_name", "cell_line_name"]


def discover_data():
    env = os.environ.get("DRUGCOMB_DATA")
    if env:
        return env
    here = Path(__file__).resolve().parent
    for cand in [here / "drugcomb_synergy_prediction_modeling.csv",
                 here.parent.parent / "drugcomb_synergy_prediction_modeling.csv",
                 here.parent.parent.parent / "drugcomb_synergy_prediction_modeling.csv"]:
        if cand.exists():
            return str(cand)
    sys.exit("Could not locate drugcomb_synergy_prediction_modeling.csv; pass --data.")


def load(data):
    use = CTX + ["drug_row", "drug_col", "drug_row_clinical_phase", "drug_col_clinical_phase",
                 "drug_row_target_name", "drug_col_target_name", "zip_synergy_label", "split"]
    df = pd.read_csv(data, usecols=use)
    df["y"] = df.zip_synergy_label.map(C2I).astype(np.int8)
    nt = lambda s: 0 if not isinstance(s, str) or s == "" else s.count(";") + 1
    df["n_targets_row"] = df.drug_row_target_name.map(nt).astype(np.int16)
    df["n_targets_col"] = df.drug_col_target_name.map(nt).astype(np.int16)
    df["has_target_row"] = (df.n_targets_row > 0).astype(np.int8)
    df["has_target_col"] = (df.n_targets_col > 0).astype(np.int8)
    ts = lambda s: frozenset() if not isinstance(s, str) or s == "" else \
        frozenset(t.strip() for t in s.split(";"))
    df["shared_target"] = np.array(
        [int(len(x & y) > 0) for x, y in zip(df.drug_row_target_name.map(ts),
                                             df.drug_col_target_name.map(ts))], dtype=np.int8)
    for c in ["drug_row_clinical_phase", "drug_col_clinical_phase"]:
        df[c] = df[c].fillna(-1).astype(np.float32)
    a, b = df.drug_row.values, df.drug_col.values
    cell = df.cell_line_name.values
    df["ord_key"] = a + "||" + b + "||" + cell
    df["rev_key"] = b + "||" + a + "||" + cell
    df["canon_key"] = np.minimum(a, b) + "||" + np.maximum(a, b) + "||" + cell
    return df


def contamination_mask(df, tr_mask, te_mask):
    """exact = same-order (pair,cell) also in train; rev = swapped counterpart in train."""
    tr_ord = set(df.loc[tr_mask, "ord_key"])
    sub = df[te_mask]
    exact = sub.ord_key.isin(tr_ord).values
    rev = sub.rev_key.isin(tr_ord).values & ~exact
    return exact, rev


def features(df, canonical):
    """Return (X, categorical_cols). canonical=True -> order-invariant drug slots."""
    if not canonical:
        cat = CTX + ["drug_row", "drug_col"]
        cols = cat + ["drug_row_clinical_phase", "drug_col_clinical_phase",
                      "n_targets_row", "n_targets_col",
                      "has_target_row", "has_target_col", "shared_target"]
        X = df[cols].copy()
    else:
        a, b = df.drug_row.values, df.drug_col.values
        swap = a > b
        X = df[CTX].copy()
        X["drug_1"] = np.where(swap, b, a)
        X["drug_2"] = np.where(swap, a, b)
        p1, p2 = df.drug_row_clinical_phase.values, df.drug_col_clinical_phase.values
        X["phase_1"] = np.where(swap, p2, p1)
        X["phase_2"] = np.where(swap, p1, p2)
        n1, n2 = df.n_targets_row.values, df.n_targets_col.values
        X["ntg_min"] = np.minimum(n1, n2)
        X["ntg_max"] = np.maximum(n1, n2)
        h1, h2 = df.has_target_row.values, df.has_target_col.values
        X["has_both"] = (h1 & h2).astype(np.int8)
        X["has_any"] = (h1 | h2).astype(np.int8)
        X["shared_target"] = df.shared_target.values
        cat = CTX + ["drug_1", "drug_2"]
    for c in cat:
        X[c] = X[c].astype("category")
    return X, cat


def train_eval(df, X, cat, tr, va, te):
    ytr = df.y.values[tr]
    cw = {i: len(ytr) / (3 * max(1, (ytr == i).sum())) for i in range(3)}
    sw = np.array([cw[v] for v in ytr])
    g = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=350,
                           learning_rate=0.1, num_leaves=64, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=2.0, max_bin=255, force_col_wise=True,
                           random_state=RNG, n_jobs=-1)
    g.fit(X[tr], ytr, sample_weight=sw, eval_set=[(X[va], df.y.values[va])],
          eval_metric="multi_logloss", categorical_feature=cat,
          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    proba = g.predict_proba(X[te])
    return proba.argmax(1), proba, df.y.values[te]


def metrics(yt, yp, pr):
    return {"n": int(len(yt)),
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro")),
            "cohen_kappa": float(cohen_kappa_score(yt, yp)),
            "macro_auc_ovr": float(roc_auc_score(yt, pr, multi_class="ovr", average="macro"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", help="audit | clean | canon | all")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "review_experiments_output"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stages = ["audit", "clean", "canon"] if "all" in args.stages else args.stages
    df = load(args.data or discover_data())
    tr = (df.split == "train").values
    va = (df.split == "validation").values
    te = (df.split == "test").values
    res = {}

    exact, rev = contamination_mask(df, tr, te)
    contam = exact | rev

    if "audit" in stages:
        a, b = df.drug_row.values, df.drug_col.values
        gg = df.groupby("canon_key")["ord_key"].nunique()
        res["audit"] = {
            "n_test": int(te.sum()),
            "rows_drug_row_lt_col": int((a < b).sum()),
            "rows_drug_row_gt_col": int((a > b).sum()),
            "canonical_keys": int(len(gg)),
            "keys_in_both_orderings": int((gg == 2).sum()),
            "test_same_order_dup_in_train": int(exact.sum()),
            "test_reversed_in_train": int(rev.sum()),
            "test_contaminated_total": int(contam.sum()),
            "pct_test_contaminated": float(100 * contam.mean())}
        print(json.dumps(res["audit"], indent=2))

    if "clean" in stages or "canon" in stages:
        for canonical in ([False, True] if "canon" in stages else [False]):
            if canonical and "canon" not in stages:
                continue
            if not canonical and "clean" not in stages:
                continue
            X, cat = features(df, canonical)
            pred, proba, yte = train_eval(df, X, cat, tr, va, te)
            tag = "canonical" if canonical else "ordered"
            res[tag] = {
                "full": metrics(yte, pred, proba),
                "clean": metrics(yte[~contam], pred[~contam], proba[~contam]),
                "contaminated_only": metrics(yte[contam], pred[contam], proba[contam])}
            for k, v in res[tag].items():
                print("%-10s %-18s n=%6d balacc=%.4f mF1=%.4f AUC=%.4f" %
                      (tag, k, v["n"], v["balanced_accuracy"], v["macro_f1"], v["macro_auc_ovr"]))

    json.dump(res, open(out / "symmetry_audit.json", "w"), indent=2)
    print("\nwrote", out / "symmetry_audit.json")


if __name__ == "__main__":
    main()
