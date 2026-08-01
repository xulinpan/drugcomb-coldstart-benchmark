"""
Multi-seed cold-start evaluation (reviewer concern 4) and leave-drug-out
decomposition into S1/S2 (reviewer concern 3).

Concern 4: the published cold-start numbers each come from a single random draw
(seed 42). Because entity frequencies are highly skewed, which studies / cell lines /
tissues / drugs land in the test set materially changes the result. This script repeats
each regime over several seeds and reports mean +/- SD and min-max.

Concern 3: the leave-drug-out test set pools two different problems. Every run also
reports S1 (exactly one unseen drug) and S2 (both drugs unseen) separately.

Usage:
  python experiments_multiseed.py --regime colddrug   --seeds 42 43 44 45 46
  python experiments_multiseed.py --regime coldcell   --seeds 42 43 44 45 46
  python experiments_multiseed.py --regime coldtissue --seeds 42 43 44 45 46
  python experiments_multiseed.py --regime coldstudy  --seeds 42 43 44 45 46
  python experiments_multiseed.py --summary          # print aggregated table

Results accumulate in review_experiments_output/multiseed.json (resumable: an existing
regime+seed entry is skipped unless --force). Conventions match train_gbm.py /
experiments_review.py: same features, class weights, LightGBM hyperparameters.
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             cohen_kappa_score, roc_auc_score, classification_report)
import lightgbm as lgb

warnings.filterwarnings("ignore")
CLASSES = ["antagonistic", "additive", "synergistic"]
C2I = {c: i for i, c in enumerate(CLASSES)}
CAT = ["study_name", "tissue_name", "cell_line_name", "drug_row", "drug_col"]
GF = CAT + ["drug_row_clinical_phase", "drug_col_clinical_phase", "n_targets_row",
            "n_targets_col", "has_target_row", "has_target_col", "shared_target"]
REGIMES = {"colddrug": ("drug", 0.12), "coldcell": ("cell_line_name", 0.15),
           "coldtissue": ("tissue_name", 0.15), "coldstudy": ("study_name", 0.15)}


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


def make_split(df, regime, seed):
    """Return (train, val, test) boolean masks + metadata, matching experiments_review.py."""
    rng = np.random.default_rng(seed)
    meta = {}
    if regime == "colddrug":
        # published protocol: 12% of unique compounds, by entity count
        drugs = pd.unique(pd.concat([df.drug_row, df.drug_col]))
        unseen = set(rng.choice(drugs, int(len(drugs) * 0.12), replace=False))
        inrow = df.drug_row.isin(unseen).values
        incol = df.drug_col.isin(unseen).values
        te = inrow | incol
        meta["n_held_out"] = len(unseen)
        meta["_n_unseen_per_row"] = (inrow.astype(int) + incol.astype(int))
    elif regime == "coldcell":
        # published protocol: 15% of unique cell lines, by entity count
        cells = df.cell_line_name.unique()
        unseen = set(rng.choice(cells, int(len(cells) * 0.15), replace=False))
        te = df.cell_line_name.isin(unseen).values
        meta["n_held_out"] = len(unseen)
        meta["held_out"] = sorted(map(str, unseen))
    else:
        # leave-study-out / leave-tissue-out: whole categories until ~15% of rows
        # (matches experiments_review.py)
        col, frac = REGIMES[regime]
        counts = df[col].value_counts()
        order = rng.permutation(counts.index.values)
        target, chosen, acc = frac * len(df), [], 0
        for g in order:
            chosen.append(g); acc += counts[g]
            if acc >= target:
                break
        te = df[col].isin(set(chosen)).values
        meta["n_held_out"] = len(chosen)
        meta["held_out"] = sorted(map(str, chosen))
    pool = ~te
    pidx = np.where(pool)[0]; rng.shuffle(pidx)
    nval = int(len(pidx) * 0.1)
    va = np.zeros(len(df), bool); va[pidx[:nval]] = True
    tr = pool.copy(); tr[pidx[:nval]] = False
    return tr, va, te, meta


def metrics(yt, yp, pr):
    rep = classification_report(yt, yp, target_names=CLASSES, output_dict=True, zero_division=0)
    try:
        auc = float(roc_auc_score(yt, pr, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    return {"n": int(len(yt)),
            "accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro")),
            "cohen_kappa": float(cohen_kappa_score(yt, yp)),
            "macro_auc_ovr": auc,
            "recall": {c: rep[c]["recall"] for c in CLASSES}}


def run_one(df, X, regime, seed):
    tr, va, te, meta = make_split(df, regime, seed)
    ytr = df.y.values[tr]
    cw = {i: len(ytr) / (3 * max(1, (ytr == i).sum())) for i in range(3)}
    sw = np.array([cw[v] for v in ytr])
    g = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=400,
                           learning_rate=0.1, num_leaves=64, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=2.0, max_bin=255, force_col_wise=True,
                           random_state=42, n_jobs=-1)
    g.fit(X[tr], ytr, sample_weight=sw, eval_set=[(X[va], df.y.values[va])],
          eval_metric="multi_logloss", categorical_feature=CAT,
          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    proba = g.predict_proba(X[te]); pred = proba.argmax(1); yte = df.y.values[te]
    out = {"regime": regime, "seed": int(seed), "n_train": int(tr.sum()),
           "n_val": int(va.sum()), "n_test": int(te.sum()),
           "n_held_out": meta["n_held_out"], "pooled": metrics(yte, pred, proba)}
    if "held_out" in meta:
        out["held_out"] = meta["held_out"]
    if regime == "colddrug":  # concern 3 decomposition
        nu = meta["_n_unseen_per_row"][te]
        for tag, m in [("S1_one_unseen", nu == 1), ("S2_both_unseen", nu == 2)]:
            if m.sum() > 0:
                out[tag] = metrics(yte[m], pred[m], proba[m])
    return out


def summarize(store):
    print("\n%-12s %-6s %-22s %-22s %-22s" % ("regime", "seeds", "bal.acc mean+/-SD [min,max]",
                                              "macro-F1 mean+/-SD", "macro-AUC mean+/-SD"))
    for regime in REGIMES:
        rows = [v for v in store.values() if v["regime"] == regime]
        if not rows:
            continue
        def agg(key):
            a = np.array([r["pooled"][key] for r in rows], float)
            return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0, a.min(), a.max()
        b = agg("balanced_accuracy"); f = agg("macro_f1"); u = agg("macro_auc_ovr")
        print("%-12s %-6d %.3f+/-%.3f [%.3f,%.3f]   %.3f+/-%.3f          %.3f+/-%.3f"
              % (regime, len(rows), b[0], b[1], b[2], b[3], f[0], f[1], u[0], u[1]))
        ns = [r["n_test"] for r in rows]
        print("%-12s %-6s n_test range %d-%d" % ("", "", min(ns), max(ns)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=list(REGIMES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--data", default=None)
    ap.add_argument("--parquet", default=None, help="optional cached feature parquet")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "review_experiments_output"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    store_p = out / "multiseed.json"
    store = json.load(open(store_p)) if store_p.exists() else {}

    if args.summary and not args.regime:
        summarize(store); return

    df = load(args.data or discover_data(), args.parquet)
    X = df[GF].copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    for seed in args.seeds:
        key = f"{args.regime}|{seed}"
        if key in store and not args.force:
            print("skip", key); continue
        t = time.time()
        store[key] = run_one(df, X, args.regime, seed)
        json.dump(store, open(store_p, "w"), indent=2)
        p = store[key]["pooled"]
        print("%-11s seed=%d n_test=%7d balacc=%.4f mF1=%.4f AUC=%.4f  (%.0fs)"
              % (args.regime, seed, store[key]["n_test"], p["balanced_accuracy"],
                 p["macro_f1"], p["macro_auc_ovr"], time.time() - t))
    summarize(store)


if __name__ == "__main__":
    main()
