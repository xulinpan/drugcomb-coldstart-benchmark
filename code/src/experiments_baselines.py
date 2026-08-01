"""
Memorization baselines (reviewer concern 5).

Question: is the LightGBM advantage over the linear baseline evidence of non-linear
drug-by-context learning, or is it recoverable from entity base rates alone?

Baselines (all fitted on the training partition only, smoothed toward the global prior
with pseudo-count s, unseen keys backing off to the coarser level):

  global        global class prior                                  (reference)
  cell          per-cell-line class prior
  pair          per-(drug pair) class prior                         -> global
  drug_cell     per-(cell line, drug) prior, averaged over the two
                compounds                                           -> cell -> global
  pair_cell     per-(cell line, drug pair) prior                    -> drug_cell -> cell -> global

Comparators:
  lgbm_depth1   additive LightGBM (max_depth=1): no feature interactions by construction
  lgbm_full     the manuscript model                                (for reference)

Because the reported models are class-weighted, hard predictions from the prior baselines
are taken after applying the same inverse-frequency weights ("balanced argmax"), so the
comparison is like-for-like. Macro-AUC is weight-independent and is reported as the
cleanest measure of ranking ability.

Usage:
  python experiments_baselines.py priors
  python experiments_baselines.py depth1
  python experiments_baselines.py all
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             cohen_kappa_score, roc_auc_score, classification_report)

warnings.filterwarnings("ignore")
RNG = 42
CLASSES = ["antagonistic", "additive", "synergistic"]
C2I = {c: i for i, c in enumerate(CLASSES)}
CAT = ["study_name", "tissue_name", "cell_line_name", "drug_row", "drug_col"]
GF = CAT + ["drug_row_clinical_phase", "drug_col_clinical_phase", "n_targets_row",
            "n_targets_col", "has_target_row", "has_target_col", "shared_target"]
SMOOTH = 20.0


def discover_data():
    env = os.environ.get("DRUGCOMB_DATA")
    if env:
        return env
    here = Path(__file__).resolve().parent
    for c in [here / "drugcomb_synergy_prediction_modeling.csv",
              here.parent.parent / "drugcomb_synergy_prediction_modeling.csv",
              here.parent.parent.parent / "drugcomb_synergy_prediction_modeling.csv"]:
        if c.exists():
            return str(c)
    sys.exit("Could not locate the modeling CSV; pass --data.")


def load(data, parquet=None):
    if parquet and Path(parquet).exists():
        return pd.read_parquet(parquet)
    use = CAT + ["drug_row_clinical_phase", "drug_col_clinical_phase",
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
    return df


def smoothed_table(tr, key_series_tr, global_p, s=SMOOTH):
    """Return dict key -> smoothed class-probability vector, fitted on training rows."""
    g = pd.crosstab(key_series_tr, tr.y.values)
    for c in range(3):
        if c not in g.columns:
            g[c] = 0
    g = g[[0, 1, 2]].astype(float)
    n = g.sum(axis=1).values[:, None]
    probs = (g.values + s * global_p[None, :]) / (n + s)
    return dict(zip(g.index.values, probs))


def lookup(keys, table, fallback):
    """fallback: (n,3) array used where key is missing."""
    out = np.array(fallback, dtype=float, copy=True)
    if out.ndim == 1:
        out = np.tile(out, (len(keys), 1))
    miss = 0
    for i, k in enumerate(keys):
        v = table.get(k)
        if v is None:
            miss += 1
        else:
            out[i] = v
    return out, miss


def evaluate(yt, proba, weights, tag, store):
    pred = (proba * weights[None, :]).argmax(1)          # balanced argmax
    raw = proba.argmax(1)                                 # unweighted argmax
    rep = classification_report(yt, pred, target_names=CLASSES, output_dict=True, zero_division=0)
    try:
        auc = float(roc_auc_score(yt, proba, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    r = {"balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
         "macro_f1": float(f1_score(yt, pred, average="macro")),
         "accuracy": float(accuracy_score(yt, pred)),
         "cohen_kappa": float(cohen_kappa_score(yt, pred)),
         "macro_auc_ovr": auc,
         "accuracy_unweighted_argmax": float(accuracy_score(yt, raw)),
         "balanced_accuracy_unweighted_argmax": float(balanced_accuracy_score(yt, raw)),
         "recall": {c: rep[c]["recall"] for c in CLASSES}}
    store[tag] = r
    print("%-14s balacc=%.4f  mF1=%.4f  AUC=%.4f  (acc=%.4f)" %
          (tag, r["balanced_accuracy"], r["macro_f1"], r["macro_auc_ovr"], r["accuracy"]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", help="priors | depth1 | all")
    ap.add_argument("--data", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "review_experiments_output"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stages = ["priors", "depth1"] if "all" in args.stages else args.stages
    store_p = out / "baselines.json"
    store = json.load(open(store_p)) if store_p.exists() else {}

    df = load(args.data or discover_data(), args.parquet)
    tr_m = (df.split == "train").values
    va_m = (df.split == "validation").values
    te_m = (df.split == "test").values
    tr, te = df[tr_m], df[te_m]
    yt = te.y.values
    counts = np.bincount(tr.y.values, minlength=3).astype(float)
    global_p = counts / counts.sum()
    weights = len(tr) / (3.0 * counts)          # same inverse-frequency weights as training
    print("global prior:", np.round(global_p, 4), " weights:", np.round(weights, 3))

    if "priors" in stages:
        pair_tr = tr.drug_row.values + "||" + tr.drug_col.values
        pair_te = te.drug_row.values + "||" + te.drug_col.values
        cell_tr, cell_te = tr.cell_line_name.values, te.cell_line_name.values
        pc_tr = cell_tr + "||" + pair_tr
        pc_te = cell_te + "||" + pair_te

        t_cell = smoothed_table(tr, pd.Series(cell_tr), global_p)
        t_pair = smoothed_table(tr, pd.Series(pair_tr), global_p)
        t_pc = smoothed_table(tr, pd.Series(pc_tr), global_p)
        # (cell, single drug) table: stack both compound slots
        cd_keys = np.concatenate([cell_tr + "||" + tr.drug_row.values,
                                  cell_tr + "||" + tr.drug_col.values])
        cd_y = np.concatenate([tr.y.values, tr.y.values])
        t_cd = smoothed_table(pd.DataFrame({"y": cd_y}), pd.Series(cd_keys), global_p)

        n = len(te)
        evaluate(yt, np.tile(global_p, (n, 1)), weights, "global", store)
        p_cell, m1 = lookup(cell_te, t_cell, global_p)
        evaluate(yt, p_cell, weights, "cell", store); store["cell"]["missing_keys"] = m1
        p_pair, m2 = lookup(pair_te, t_pair, global_p)
        evaluate(yt, p_pair, weights, "pair", store); store["pair"]["missing_keys"] = m2
        a, _ = lookup(cell_te + "||" + te.drug_row.values, t_cd, p_cell)
        b, _ = lookup(cell_te + "||" + te.drug_col.values, t_cd, p_cell)
        p_dc = (a + b) / 2.0
        evaluate(yt, p_dc, weights, "drug_cell", store)
        p_pc, m4 = lookup(pc_te, t_pc, p_dc)
        evaluate(yt, p_pc, weights, "pair_cell", store); store["pair_cell"]["missing_keys"] = m4
        json.dump(store, open(store_p, "w"), indent=2)

    if "depth1" in stages:
        import lightgbm as lgb
        X = df[GF].copy()
        for c in CAT:
            X[c] = X[c].astype("category")
        ytr = df.y.values[tr_m]
        sw = np.array([weights[v] for v in ytr])
        t0 = time.time()
        g = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=600,
                               learning_rate=0.1, num_leaves=2, max_depth=1,
                               min_child_samples=100, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=2.0, max_bin=255,
                               force_col_wise=True, random_state=RNG, n_jobs=-1)
        g.fit(X[tr_m], ytr, sample_weight=sw, eval_set=[(X[va_m], df.y.values[va_m])],
              eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
        proba = g.predict_proba(X[te_m])
        pred = proba.argmax(1)   # already class-weighted during training
        rep = classification_report(yt, pred, target_names=CLASSES, output_dict=True, zero_division=0)
        store["lgbm_depth1"] = {
            "balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
            "macro_f1": float(f1_score(yt, pred, average="macro")),
            "accuracy": float(accuracy_score(yt, pred)),
            "cohen_kappa": float(cohen_kappa_score(yt, pred)),
            "macro_auc_ovr": float(roc_auc_score(yt, proba, multi_class="ovr", average="macro")),
            "best_iteration": int(g.best_iteration_ or 600),
            "recall": {c: rep[c]["recall"] for c in CLASSES}}
        r = store["lgbm_depth1"]
        print("%-14s balacc=%.4f  mF1=%.4f  AUC=%.4f  (%.0fs, %d trees)" %
              ("lgbm_depth1", r["balanced_accuracy"], r["macro_f1"], r["macro_auc_ovr"],
               time.time() - t0, r["best_iteration"]))
        json.dump(store, open(store_p, "w"), indent=2)

    print("\nwrote", store_p)


if __name__ == "__main__":
    main()
