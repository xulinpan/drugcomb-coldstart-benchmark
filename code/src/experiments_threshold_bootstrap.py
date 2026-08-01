"""
Threshold sensitivity (reviewer concern 7) and dependence-aware confidence intervals
(reviewer concern 6).

Stages
------
threshold   Re-derive the three-class endpoint at |ZIP| cut-offs of 5, 10, 15 and 20,
            retrain, and report how performance and class balance move.

nearcut     Near-threshold exclusion: with the primary (+/-10) model, drop test rows whose
            ZIP score lies within a margin of the cut and re-evaluate. Rows adjacent to the
            boundary are intrinsically ambiguous, so this quantifies how much residual error
            is attributable to label noise.

bootstrap   Confidence intervals that respect sample dependence. Rows are not independent:
            they share cell lines and compounds. Alongside the i.i.d. row bootstrap used in
            the primary analysis we report a cluster bootstrap that resamples whole cell
            lines (and, separately, whole compounds), which widens the intervals
            appropriately.

Usage
-----
  python experiments_threshold_bootstrap.py threshold --cuts 5 15 20
  python experiments_threshold_bootstrap.py nearcut
  python experiments_threshold_bootstrap.py bootstrap
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
CAT = ["study_name", "tissue_name", "cell_line_name", "drug_row", "drug_col"]
GF = CAT + ["drug_row_clinical_phase", "drug_col_clinical_phase", "n_targets_row",
            "n_targets_col", "has_target_row", "has_target_col", "shared_target"]


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
    use = ["block_id"] + CAT + ["drug_row_clinical_phase", "drug_col_clinical_phase",
                                "drug_row_target_name", "drug_col_target_name",
                                "synergy_zip", "zip_synergy_label", "split"]
    df = pd.read_csv(data, usecols=use)
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
    return df.drop(columns=["drug_row_target_name", "drug_col_target_name"])


def label_at(zip_score, cut):
    """0 antagonistic, 1 additive, 2 synergistic."""
    y = np.ones(len(zip_score), dtype=np.int8)
    y[zip_score <= -cut] = 0
    y[zip_score >= cut] = 2
    return y


def metrics(yt, yp, pr=None):
    rep = classification_report(yt, yp, target_names=CLASSES, output_dict=True, zero_division=0)
    r = {"n": int(len(yt)),
         "accuracy": float(accuracy_score(yt, yp)),
         "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
         "macro_f1": float(f1_score(yt, yp, average="macro")),
         "cohen_kappa": float(cohen_kappa_score(yt, yp)),
         "recall": {c: rep[c]["recall"] for c in CLASSES}}
    if pr is not None:
        try:
            r["macro_auc_ovr"] = float(roc_auc_score(yt, pr, multi_class="ovr", average="macro"))
        except Exception:
            r["macro_auc_ovr"] = float("nan")
    return r


def train_at(df, y, seed=RNG):
    import lightgbm as lgb
    X = df[GF].copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    tr = (df.split == "train").values
    va = (df.split == "validation").values
    te = (df.split == "test").values
    ytr = y[tr]
    counts = np.bincount(ytr, minlength=3).astype(float)
    w = len(ytr) / (3.0 * np.maximum(counts, 1))
    sw = np.array([w[v] for v in ytr])
    g = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=350,
                           learning_rate=0.1, num_leaves=64, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=2.0, max_bin=255, force_col_wise=True,
                           random_state=seed, n_jobs=-1)
    g.fit(X[tr], ytr, sample_weight=sw, eval_set=[(X[va], y[va])],
          eval_metric="multi_logloss", categorical_feature=CAT,
          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    proba = g.predict_proba(X[te])
    return proba.argmax(1), proba, y[te]


def cluster_bootstrap(yt, pred, clusters, n_boot, rng):
    """Resample whole clusters with replacement; return percentile CI for key metrics."""
    uniq = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in uniq}
    bal, mf1, acc = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        yb, pb = yt[idx], pred[idx]
        if len(np.unique(yb)) < 2:
            continue
        bal.append(balanced_accuracy_score(yb, pb))
        mf1.append(f1_score(yb, pb, average="macro"))
        acc.append(accuracy_score(yb, pb))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"balanced_accuracy_ci": ci(bal), "macro_f1_ci": ci(mf1),
            "accuracy_ci": ci(acc), "n_boot": len(bal), "n_clusters": int(len(uniq))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", help="threshold | nearcut | bootstrap")
    ap.add_argument("--cuts", type=float, nargs="+", default=[5, 15, 20])
    ap.add_argument("--margins", type=float, nargs="+", default=[1, 2, 5])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--data", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--preds", default=None, help="npz with yte,pred,proba for the +/-10 model")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "review_experiments_output"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    store_p = out / "threshold_bootstrap.json"
    store = json.load(open(store_p)) if store_p.exists() else {}
    df = load(args.data or discover_data(), args.parquet)
    te = (df.split == "test").values
    zip_te = df.synergy_zip.values[te]

    if "threshold" in args.stages:
        store.setdefault("threshold", {})
        for cut in args.cuts:
            y = label_at(df.synergy_zip.values, cut)
            share = np.bincount(y, minlength=3) / len(y)
            t0 = time.time()
            pred, proba, yt = train_at(df, y)
            m = metrics(yt, pred, proba)
            m["class_share_full_dataset"] = share.tolist()
            store["threshold"][str(int(cut))] = m
            print("cut=+/-%-3g share(A/Ad/S)=%.3f/%.3f/%.3f  balacc=%.4f mF1=%.4f AUC=%.4f (%.0fs)"
                  % (cut, share[0], share[1], share[2], m["balanced_accuracy"],
                     m["macro_f1"], m["macro_auc_ovr"], time.time() - t0))
            json.dump(store, open(store_p, "w"), indent=2)

    if "nearcut" in args.stages:
        d = np.load(args.preds or "/tmp/preds.npz")
        yt, pred, proba = d["yte"], d["pred"], d["proba"]
        assert len(yt) == te.sum(), "prediction file does not match the test partition"
        store["nearcut"] = {"full": metrics(yt, pred, proba)}
        for mg in args.margins:
            keep = np.abs(np.abs(zip_te) - 10.0) > mg
            m = metrics(yt[keep], pred[keep], proba[keep])
            m["excluded"] = int((~keep).sum())
            m["excluded_pct"] = float(100 * (~keep).mean())
            store["nearcut"][f"margin_{mg:g}"] = m
            print("margin +/-%-3g excluded=%6d (%4.1f%%)  balacc=%.4f  mF1=%.4f"
                  % (mg, m["excluded"], m["excluded_pct"], m["balanced_accuracy"], m["macro_f1"]))
        json.dump(store, open(store_p, "w"), indent=2)

    if "bootstrap" in args.stages:
        d = np.load(args.preds or "/tmp/preds.npz")
        yt, pred = d["yte"], d["pred"]
        rng = np.random.default_rng(RNG)
        store["bootstrap"] = {}
        # i.i.d. row bootstrap (as in the primary analysis)
        n = len(yt); bal = []; mf1 = []
        for _ in range(args.n_boot):
            i = rng.integers(0, n, n)
            bal.append(balanced_accuracy_score(yt[i], pred[i]))
            mf1.append(f1_score(yt[i], pred[i], average="macro"))
        ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        store["bootstrap"]["row_iid"] = {"balanced_accuracy_ci": ci(bal),
                                         "macro_f1_ci": ci(mf1), "n_boot": args.n_boot}
        print("row i.i.d.      balacc CI=[%.4f,%.4f]" % tuple(ci(bal)))
        for name, col in [("cell_line", "cell_line_name"), ("drug_row", "drug_row")]:
            cl = df[col].values[te]
            r = cluster_bootstrap(yt, pred, cl, args.n_boot, rng)
            store["bootstrap"][f"cluster_{name}"] = r
            print("cluster:%-10s balacc CI=[%.4f,%.4f]  (%d clusters)"
                  % (name, r["balanced_accuracy_ci"][0], r["balanced_accuracy_ci"][1],
                     r["n_clusters"]))
        json.dump(store, open(store_p, "w"), indent=2)

    print("\nwrote", store_p)


if __name__ == "__main__":
    main()
