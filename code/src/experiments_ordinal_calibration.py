"""
Ordinal error structure (reviewer 2, comment 3) and calibration assessment
(reviewer 2, comment 4).

The three classes are ordinal (antagonistic < additive < synergistic), so the claim that
errors are mostly interacting-vs-additive rather than synergistic-vs-antagonistic needs a
distance-sensitive statistic, not a visual reading of the confusion matrix. And calibration
must be reported for every model with a quantitative basis, not described qualitatively.

Stages
------
lr            Refit the logistic baseline and cache its test probabilities (needed because
              the ordinal and calibration statistics require predicted distributions).
ordinal       Ranked probability score (RPS), mean absolute ordinal distance, and the split
              of errors into distance-1 and distance-2, for every model.
calibration   Expected and maximum calibration error (top-label and per-class one-vs-rest)
              and Brier scores, for every model, plus reliability-curve data for plotting.

RPS for K ordinal classes with predicted distribution p and true class y:
    RPS = 1/(K-1) * sum_k ( CDF_p(k) - 1[y <= k] )^2 ,   lower is better.

Usage
-----
  python experiments_ordinal_calibration.py lr
  python experiments_ordinal_calibration.py ordinal calibration
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")
RNG = 42
CLASSES = ["antagonistic", "additive", "synergistic"]
C2I = {c: i for i, c in enumerate(CLASSES)}
CAT = ["study_name", "tissue_name", "cell_line_name", "drug_row", "drug_col"]
NUMP = ["drug_row_clinical_phase", "drug_col_clinical_phase"]
EXTRA = ["n_targets_row", "n_targets_col", "has_target_row", "has_target_col", "shared_target"]
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
    use = CAT + NUMP + ["drug_row_target_name", "drug_col_target_name",
                        "zip_synergy_label", "split"]
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
    for c in NUMP:
        df[c] = df[c].fillna(-1).astype(np.float32)
    return df


# ------------------------------------------------------------------ logistic baseline
def fit_logistic(df, cache):
    """Replicates train_lr.py: frequency + smoothed mean-target encoding, standardized."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    tr_m = (df.split == "train").values
    te_m = (df.split == "test").values
    tr = df[tr_m]
    for i, c in enumerate(CLASSES):
        df[f"is_{c}"] = (df.y == i).astype(np.int8)
    tr = df[tr_m]
    feats, X = [], {}
    for col in CAT:
        fm = tr[col].value_counts(normalize=True)
        X[f"freq_{col}"] = df[col].map(fm).fillna(0.0).values.astype(np.float32)
        feats.append(f"freq_{col}")
        for c in CLASSES:
            oh = f"is_{c}"
            g = tr[oh].mean()
            a = tr.groupby(col)[oh].agg(["mean", "count"])
            e = (a["mean"] * a["count"] + g * SMOOTH) / (a["count"] + SMOOTH)
            nm = f"te_{col}_{c}"
            X[nm] = df[col].map(e).fillna(g).values.astype(np.float32)
            feats.append(nm)
    for col in NUMP + EXTRA:
        X[col] = df[col].values.astype(np.float32)
        feats.append(col)
    Xdf = pd.DataFrame(X)
    sc = StandardScaler().fit(Xdf.values[tr_m])
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                            multi_class="multinomial", n_jobs=-1, random_state=RNG)
    t0 = time.time()
    lr.fit(sc.transform(Xdf.values[tr_m]), df.y.values[tr_m])
    proba = lr.predict_proba(sc.transform(Xdf.values[te_m]))
    np.savez(cache, proba=proba, yte=df.y.values[te_m])
    print("logistic refit in %.0fs -> %s" % (time.time() - t0, cache))
    return proba


# ------------------------------------------------------------------ baselines / metrics
def memorization_proba(df):
    """Smoothed per-(cell line, drug) prior averaged over the two compound slots."""
    tr_m = (df.split == "train").values
    te_m = (df.split == "test").values
    tr, te = df[tr_m], df[te_m]
    counts = np.bincount(tr.y.values, minlength=3).astype(float)
    gp = counts / counts.sum()

    def table(keys, ys):
        g = pd.crosstab(pd.Series(keys), ys)
        for c in range(3):
            if c not in g.columns:
                g[c] = 0
        g = g[[0, 1, 2]].astype(float)
        n = g.sum(axis=1).values[:, None]
        return dict(zip(g.index.values, (g.values + SMOOTH * gp[None, :]) / (n + SMOOTH)))

    cell_t = table(tr.cell_line_name.values, tr.y.values)
    cd_keys = np.concatenate([tr.cell_line_name.values + "||" + tr.drug_row.values,
                              tr.cell_line_name.values + "||" + tr.drug_col.values])
    cd_t = table(cd_keys, np.concatenate([tr.y.values, tr.y.values]))

    def lk(keys, tab, fb):
        out = np.array(fb, dtype=float, copy=True)
        if out.ndim == 1:
            out = np.tile(out, (len(keys), 1))
        for i, k in enumerate(keys):
            v = tab.get(k)
            if v is not None:
                out[i] = v
        return out
    p_cell = lk(te.cell_line_name.values, cell_t, gp)
    a = lk(te.cell_line_name.values + "||" + te.drug_row.values, cd_t, p_cell)
    b = lk(te.cell_line_name.values + "||" + te.drug_col.values, cd_t, p_cell)
    return (a + b) / 2.0, np.tile(gp, (len(te), 1))


def rps(y, proba):
    cdf_p = np.cumsum(proba, axis=1)
    cdf_t = (np.arange(3)[None, :] >= y[:, None]).astype(float)
    return float(((cdf_p - cdf_t) ** 2).sum(axis=1).mean() / 2.0)


def ordinal_stats(y, proba, weights=None):
    pred = (proba * weights[None, :]).argmax(1) if weights is not None else proba.argmax(1)
    d = np.abs(pred - y)
    err = d > 0
    return {"rps": rps(y, proba),
            "mean_abs_ordinal_distance": float(d.mean()),
            "n_errors": int(err.sum()),
            "err_distance1": int((d == 1).sum()),
            "err_distance2": int((d == 2).sum()),
            "pct_of_errors_distance2": float(100 * (d == 2).sum() / max(1, err.sum())),
            "pct_of_all_rows_distance2": float(100 * (d == 2).mean())}


def ece_mce(y_true_bin, p, n_bins=15):
    """Binary (one-vs-rest) calibration error with uniform bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0; mce = 0.0; n = len(p)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        gap = abs(y_true_bin[m].mean() - p[m].mean())
        ece += m.sum() / n * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def top_label_ece(y, proba, n_bins=15):
    conf = proba.max(1); pred = proba.argmax(1); correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0; n = len(conf)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def calibration_block(y, proba, n_bins=15):
    out = {"top_label_ece": top_label_ece(y, proba, n_bins),
           "brier_multiclass": float(((proba - np.eye(3)[y]) ** 2).sum(axis=1).mean()),
           "per_class": {}}
    for i, c in enumerate(CLASSES):
        yb = (y == i).astype(float)
        e, m = ece_mce(yb, proba[:, i], n_bins)
        out["per_class"][c] = {"ece": e, "mce": m,
                               "brier": float(brier_score_loss(yb, proba[:, i])),
                               "mean_predicted": float(proba[:, i].mean()),
                               "observed_rate": float(yb.mean())}
    return out


def reliability_curve(y, proba, n_bins=10):
    """Quantile-binned reliability data per class, for plotting."""
    out = {}
    for i, c in enumerate(CLASSES):
        yb = (y == i).astype(float); p = proba[:, i]
        qs = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        idx = np.clip(np.digitize(p, qs[1:-1], right=False), 0, len(qs) - 2)
        mp, fp = [], []
        for b in range(len(qs) - 1):
            m = idx == b
            if m.sum() > 0:
                mp.append(float(p[m].mean())); fp.append(float(yb[m].mean()))
        out[c] = {"mean_pred": mp, "frac_pos": fp}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", help="lr | ordinal | calibration")
    ap.add_argument("--data", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--gbm-preds", default="/tmp/preds.npz")
    ap.add_argument("--lr-cache", default="/tmp/lr_proba.npz")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "review_experiments_output"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    store_p = out / "ordinal_calibration.json"
    store = json.load(open(store_p)) if store_p.exists() else {}
    df = load(args.data or discover_data(), args.parquet)

    if "lr" in args.stages:
        fit_logistic(df, args.lr_cache); return

    g = np.load(args.gbm_preds); y = g["yte"]; p_gbm = g["proba"]
    models = {"lightgbm": (p_gbm, None)}
    if Path(args.lr_cache).exists():
        lrp = np.load(args.lr_cache)
        assert np.array_equal(lrp["yte"], y), "logistic cache does not match the test partition"
        models["logistic"] = (lrp["proba"], None)
    tr = df[(df.split == "train").values]
    counts = np.bincount(tr.y.values, minlength=3).astype(float)
    w = len(tr) / (3.0 * counts)
    p_mem, p_glob = memorization_proba(df)
    models["memorization_cell_drug"] = (p_mem, w)   # weighted argmax, as in the baselines table
    models["global_prior"] = (p_glob, w)

    if "ordinal" in args.stages:
        store["ordinal"] = {k: ordinal_stats(y, p, wt) for k, (p, wt) in models.items()}
        print("%-24s %8s %8s %10s %12s" % ("model", "RPS", "MAOD", "err d=2", "%err d=2"))
        for k, v in store["ordinal"].items():
            print("%-24s %8.4f %8.4f %10d %11.1f%%" %
                  (k, v["rps"], v["mean_abs_ordinal_distance"], v["err_distance2"],
                   v["pct_of_errors_distance2"]))
        json.dump(store, open(store_p, "w"), indent=2)

    if "calibration" in args.stages:
        store["calibration"] = {"n_bins": args.bins}
        for k, (p, _) in models.items():
            store["calibration"][k] = calibration_block(y, p, args.bins)
            store["calibration"][k]["reliability"] = reliability_curve(y, p)
        print("\n%-24s %10s %10s %10s %10s" % ("model", "topECE", "Brier", "ECE syn", "ECE ant"))
        for k in models:
            c = store["calibration"][k]
            print("%-24s %10.4f %10.4f %10.4f %10.4f" %
                  (k, c["top_label_ece"], c["brier_multiclass"],
                   c["per_class"]["synergistic"]["ece"], c["per_class"]["antagonistic"]["ece"]))
        json.dump(store, open(store_p, "w"), indent=2)

    print("\nwrote", store_p)


if __name__ == "__main__":
    main()
