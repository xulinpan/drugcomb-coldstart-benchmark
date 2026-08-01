"""
Review-driven robustness experiments for the DrugComb cold-start benchmark.

Adds three analyses requested by peer review, all using ONLY the identity-and-context
feature set (no molecular structure, no SMILES required):

  nostudy      No-study ablation on the standard split. Trains the LightGBM model with
               and without the `study_name` feature and compares test performance. Tests
               whether the reported numbers depend on the study confounder (review point M3).

  leavestudy   Leave-study-out. Holds out a set of whole studies (~15% of rows by default)
               as the test set; trains on the remaining studies. Measures cross-study
               transfer / platform generalization (review point M3).

  leavetissue  Leave-tissue-out. Holds out a set of whole tissues (~15% of rows) as the
               test set; disentangles a clean tissue effect from the cell-line effect in
               the cold-cell regime (review point M4).

Usage:
  python experiments_review.py prep                      # build processed.parquet (once)
  python experiments_review.py nostudy leavestudy leavetissue
  python experiments_review.py all

Paths are configurable with --data / --out or the env vars DRUGCOMB_DATA / DRUGCOMB_OUT.
Outputs: one JSON per stage in --out, plus review_experiments_summary.md.

Matches the conventions of train_gbm.py / analysis.py (same features, class weighting,
LightGBM hyperparameters, fixed seed 42).
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, roc_auc_score, cohen_kappa_score,
                             classification_report)
import lightgbm as lgb

warnings.filterwarnings("ignore")
RNG = 42

CLASSES = ["antagonistic", "additive", "synergistic"]
C2I = {c: i for i, c in enumerate(CLASSES)}
CAT = ["study_name", "tissue_name", "cell_line_name", "drug_row", "drug_col"]
NUMP = ["drug_row_clinical_phase", "drug_col_clinical_phase"]
EXTRA = ["n_targets_row", "n_targets_col", "has_target_row", "has_target_col", "shared_target"]
GFEAT = CAT + NUMP + EXTRA


# ---------------------------------------------------------------- paths
def discover_data():
    env = os.environ.get("DRUGCOMB_DATA")
    if env:
        return env
    here = Path(__file__).resolve().parent
    for cand in [
        here / "drugcomb_synergy_prediction_modeling.csv",
        here.parent.parent / "drugcomb_synergy_prediction_modeling.csv",          # modeling/
        here.parent.parent.parent / "drugcomb_synergy_prediction_modeling.csv",
    ]:
        if cand.exists():
            return str(cand)
    return str(here / "drugcomb_synergy_prediction_modeling.csv")


# ---------------------------------------------------------------- feature build
def build_parquet(data, out):
    pq = out / "processed.parquet"
    if pq.exists():
        print(f"prep skip ({pq} exists)")
        return pq
    t0 = time.time()
    use = CAT + NUMP + ["drug_row_target_name", "drug_col_target_name",
                        "zip_synergy_label", "split"]
    df = pd.read_csv(data, usecols=use)
    df["y"] = df["zip_synergy_label"].map(C2I).astype(np.int8)

    def n_targets(s):
        return 0 if not isinstance(s, str) or s == "" else s.count(";") + 1

    df["n_targets_row"] = df["drug_row_target_name"].map(n_targets).astype(np.int16)
    df["n_targets_col"] = df["drug_col_target_name"].map(n_targets).astype(np.int16)
    df["has_target_row"] = (df["n_targets_row"] > 0).astype(np.int8)
    df["has_target_col"] = (df["n_targets_col"] > 0).astype(np.int8)

    def to_set(s):
        return frozenset() if not isinstance(s, str) or s == "" else \
            frozenset(t.strip() for t in s.split(";"))
    sr = df["drug_row_target_name"].map(to_set)
    sc = df["drug_col_target_name"].map(to_set)
    df["shared_target"] = np.array([int(len(a & b) > 0) for a, b in zip(sr, sc)], dtype=np.int8)
    for c in NUMP:
        df[c] = df[c].fillna(-1).astype(np.float32)
    keep = GFEAT + ["y", "split"]
    df[keep].to_parquet(pq, index=False)
    print(f"prep: saved {pq} rows={len(df):,} in {time.time()-t0:.1f}s")
    return pq


def load(out):
    return pd.read_parquet(out / "processed.parquet")


# ---------------------------------------------------------------- model / metrics
def train_gbm(df, m_tr, m_va, m_te, feats, n_est=400):
    ytr, yva, yte = df.y.values[m_tr], df.y.values[m_va], df.y.values[m_te]
    cats = [c for c in CAT if c in feats]
    X = df[feats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    cw = {i: len(ytr) / (3 * max(1, np.sum(ytr == i))) for i in range(3)}
    sw = np.array([cw[v] for v in ytr])
    g = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=n_est,
                           learning_rate=0.1, num_leaves=64, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           reg_lambda=2.0, max_bin=255, force_col_wise=True,
                           random_state=RNG, n_jobs=-1)
    g.fit(X[m_tr], ytr, sample_weight=sw, eval_set=[(X[m_va], yva)],
          eval_metric="multi_logloss", categorical_feature=cats,
          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    proba = g.predict_proba(X[m_te])
    return g, proba.argmax(1), proba, yte


def metrics_block(yt, yp, yproba):
    rep = classification_report(yt, yp, target_names=CLASSES, output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro")),
            "weighted_f1": float(f1_score(yt, yp, average="weighted")),
            "cohen_kappa": float(cohen_kappa_score(yt, yp)),
            "macro_auc_ovr": float(roc_auc_score(yt, yproba, multi_class="ovr", average="macro")),
            "per_class": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                              "f1": rep[c]["f1-score"], "support": rep[c]["support"]} for c in CLASSES},
            "confusion_matrix": confusion_matrix(yt, yp).tolist()}


def _holdout_by_group(df, col, frac, rng):
    """Pick whole categories of `col` (in random order) until ~frac of rows are held out."""
    counts = df[col].value_counts()
    order = rng.permutation(counts.index.values)
    target = frac * len(df)
    chosen, acc = [], 0
    for g in order:
        chosen.append(g); acc += counts[g]
        if acc >= target:
            break
    test = df[col].isin(set(chosen)).values
    return set(chosen), test


# ---------------------------------------------------------------- stages
def stage_nostudy(df, out):
    m_tr = (df.split == "train").values
    m_va = (df.split == "validation").values
    m_te = (df.split == "test").values
    res = {"split": "standard (grouped)", "n_test": int(m_te.sum())}
    for tag, feats in [("with_study", GFEAT), ("no_study", [f for f in GFEAT if f != "study_name"])]:
        _, pred, proba, yte = train_gbm(df, m_tr, m_va, m_te, feats)
        res[tag] = metrics_block(yte, pred, proba)
        print(f"  {tag}: balacc={res[tag]['balanced_accuracy']:.4f} mf1={res[tag]['macro_f1']:.4f}")
    res["delta_balanced_accuracy"] = res["no_study"]["balanced_accuracy"] - res["with_study"]["balanced_accuracy"]
    res["delta_macro_f1"] = res["no_study"]["macro_f1"] - res["with_study"]["macro_f1"]
    json.dump(res, open(out / "nostudy.json", "w"), indent=2)
    return res


def _leaveout(df, col, out, frac=0.15):
    rng = np.random.default_rng(RNG)
    chosen, te = _holdout_by_group(df, col, frac, rng)
    pool = ~te
    pidx = np.where(pool)[0]; rng.shuffle(pidx)
    nval = int(len(pidx) * 0.1)
    va = np.zeros(len(df), bool); va[pidx[:nval]] = True
    tr = pool.copy(); tr[pidx[:nval]] = False
    # the held-out column is uninformative for unseen test categories; keep it in the
    # feature set so LightGBM treats unseen levels as missing (honest leave-X-out).
    _, pred, proba, yte = train_gbm(df, tr, va, te, GFEAT)
    res = {"split": f"leave-{col}-out", "held_out": sorted(map(str, chosen)),
           "n_held_out": len(chosen), "n_train": int(tr.sum()), "n_val": int(va.sum()),
           "n_test": int(te.sum()), "metrics": metrics_block(yte, pred, proba)}
    m = res["metrics"]
    print(f"  leave-{col}-out: n_test={res['n_test']:,} balacc={m['balanced_accuracy']:.4f} mf1={m['macro_f1']:.4f}")
    return res


def stage_leavestudy(df, out):
    res = _leaveout(df, "study_name", out)
    json.dump(res, open(out / "leavestudy.json", "w"), indent=2)
    return res


def stage_leavetissue(df, out):
    res = _leaveout(df, "tissue_name", out)
    json.dump(res, open(out / "leavetissue.json", "w"), indent=2)
    return res


# ---------------------------------------------------------------- summary
def write_summary(out):
    rows = []
    def grab(fname, label, key="metrics"):
        p = out / fname
        if not p.exists():
            return
        d = json.load(open(p))
        m = d.get(key, d)
        rows.append((label, m["balanced_accuracy"], m["macro_f1"], m["macro_auc_ovr"],
                     m["per_class"]["antagonistic"]["recall"],
                     m["per_class"]["additive"]["recall"],
                     m["per_class"]["synergistic"]["recall"]))
    ns = out / "nostudy.json"
    if ns.exists():
        d = json.load(open(ns))
        for tag, label in [("with_study", "Standard, with study"), ("no_study", "Standard, no study")]:
            m = d[tag]
            rows.append((label, m["balanced_accuracy"], m["macro_f1"], m["macro_auc_ovr"],
                         m["per_class"]["antagonistic"]["recall"], m["per_class"]["additive"]["recall"],
                         m["per_class"]["synergistic"]["recall"]))
    grab("leavestudy.json", "Leave-study-out")
    grab("leavetissue.json", "Leave-tissue-out")
    if not rows:
        return
    lines = ["# Review experiments — summary", "",
             "| Regime | Bal. acc. | Macro-F1 | Macro-AUC | Rec. Antag. | Rec. Add. | Rec. Syn. |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(*r))
    (out / "review_experiments_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+",
                    help="prep | nostudy | leavestudy | leavetissue | all")
    ap.add_argument("--data", default=discover_data())
    ap.add_argument("--out", default=os.environ.get("DRUGCOMB_OUT",
                    str(Path(__file__).resolve().parent / "review_experiments_output")))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stages = ["prep", "nostudy", "leavestudy", "leavetissue"] if "all" in args.stages else args.stages

    if "prep" in stages:
        build_parquet(args.data, out)
        stages = [s for s in stages if s != "prep"]
    if not stages:
        return
    df = load(out)
    table = {"nostudy": stage_nostudy, "leavestudy": stage_leavestudy, "leavetissue": stage_leavetissue}
    for s in stages:
        t = time.time(); print(f"[{s}] start"); table[s](df, out); print(f"[{s}] {time.time()-t:.1f}s")
    write_summary(out)


if __name__ == "__main__":
    main()
