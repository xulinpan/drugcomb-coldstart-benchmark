"""
Molecular (Morgan-fingerprint) baseline for the DrugComb cold-start benchmark
(peer-review point M2).

Adds chemical-structure features on top of / instead of the identity-and-context
descriptors, and evaluates on the SAME standard / leave-drug-out / leave-cell-line-out
splits so the structure model can be compared directly with the identity-only model.

Requires:
  * RDKit            pip install rdkit
  * lightgbm, pandas, numpy, scikit-learn
  * a drug -> SMILES table (CSV with columns: drug, smiles). DrugComb publishes drug
    annotations with SMILES; or build one with fetch_drug_smiles.py. Drugs without a
    SMILES get an all-zero fingerprint and are flagged.

Feature modes (--mode):
  struct        ECFP4 fingerprint of drug A + drug B, plus Tanimoto similarity and
                shared/!set bit counts.  (structure only)
  struct_ctx    struct features PLUS the context categoricals (cell line, tissue, study)
                and the clinical-phase / target descriptors.  (recommended comparator)

Usage:
  python fingerprint_baseline.py --smiles drug_smiles.csv --mode struct_ctx \
         --regime standard --nbits 256 --radius 2
  python fingerprint_baseline.py --smiles drug_smiles.csv --regime colddrug
  python fingerprint_baseline.py --smiles drug_smiles.csv --regime coldcell

Writes fingerprint_<mode>_<regime>.json with a metrics block comparable to the
identity-only results in metrics_gbm.json / colddrug.json / coldcell.json.
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, roc_auc_score, cohen_kappa_score,
                             average_precision_score, classification_report)
from sklearn.preprocessing import label_binarize
import lightgbm as lgb

warnings.filterwarnings("ignore")
RNG = 42
CLASSES = ["antagonistic", "additive", "synergistic"]
C2I = {c: i for i, c in enumerate(CLASSES)}
CTX_CAT = ["study_name", "tissue_name", "cell_line_name"]
CTX_NUM = ["drug_row_clinical_phase", "drug_col_clinical_phase",
           "n_targets_row", "n_targets_col", "has_target_row", "has_target_col", "shared_target"]


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


def build_fingerprints(drugs, smiles_map, nbits, radius):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = {}
    n_ok = 0
    for d in drugs:
        smi = smiles_map.get(d)
        arr = np.zeros(nbits, dtype=np.uint8)
        if isinstance(smi, str) and smi:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
                arr = np.frombuffer(bytes(bv.ToBitString(), "ascii"), dtype=np.uint8) - ord("0")
                n_ok += 1
        fps[d] = arr
    print(f"  fingerprints: {n_ok}/{len(drugs)} drugs resolved to a valid structure")
    return fps


def pair_features(df, fps, nbits):
    A = np.stack([fps[d] for d in df.drug_row.values]).astype(np.float32)
    B = np.stack([fps[d] for d in df.drug_col.values]).astype(np.float32)
    inter = (A * B).sum(1)
    union = ((A + B) > 0).sum(1)
    tani = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    extra = np.column_stack([tani, A.sum(1), B.sum(1), inter]).astype(np.float32)
    cols = ([f"fpA_{i}" for i in range(nbits)] + [f"fpB_{i}" for i in range(nbits)] +
            ["tanimoto", "nbitsA", "nbitsB", "shared_bits"])
    return pd.DataFrame(np.hstack([A, B, extra]), columns=cols, index=df.index)


def make_splits(df, regime):
    rng = np.random.default_rng(RNG)
    if regime == "standard":
        return (df.split == "train").values, (df.split == "validation").values, (df.split == "test").values
    if regime == "colddrug":
        drugs = pd.unique(pd.concat([df.drug_row, df.drug_col]))
        unseen = set(rng.choice(drugs, int(len(drugs) * 0.12), replace=False))
        te = (df.drug_row.isin(unseen) | df.drug_col.isin(unseen)).values
    elif regime == "coldcell":
        cells = df.cell_line_name.unique()
        unseen = set(rng.choice(cells, int(len(cells) * 0.15), replace=False))
        te = df.cell_line_name.isin(unseen).values
    else:
        sys.exit(f"unknown regime {regime}")
    pool = ~te; pidx = np.where(pool)[0]; rng.shuffle(pidx)
    nval = int(len(pidx) * 0.1)
    va = np.zeros(len(df), bool); va[pidx[:nval]] = True
    tr = pool.copy(); tr[pidx[:nval]] = False
    return tr, va, te


def metrics_block(yt, yp, yproba):
    rep = classification_report(yt, yp, target_names=CLASSES, output_dict=True, zero_division=0)
    yb = label_binarize(yt, classes=[0, 1, 2])
    auprc = {c: float(average_precision_score(yb[:, i], yproba[:, i])) for i, c in enumerate(CLASSES)}
    return {"accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro")),
            "cohen_kappa": float(cohen_kappa_score(yt, yp)),
            "macro_auc_ovr": float(roc_auc_score(yt, yproba, multi_class="ovr", average="macro")),
            "auprc": auprc,
            "per_class": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                              "f1": rep[c]["f1-score"], "support": rep[c]["support"]} for c in CLASSES},
            "confusion_matrix": confusion_matrix(yt, yp).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--smiles", required=True, help="CSV with columns drug,smiles")
    ap.add_argument("--mode", choices=["struct", "struct_ctx"], default="struct_ctx")
    ap.add_argument("--regime", choices=["standard", "colddrug", "coldcell"], default="standard")
    ap.add_argument("--nbits", type=int, default=256)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "review_experiments_output"))
    args = ap.parse_args()
    data = args.data or discover_data()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    use = ["study_name", "tissue_name", "cell_line_name", "drug_row", "drug_col",
           "drug_row_clinical_phase", "drug_col_clinical_phase",
           "drug_row_target_name", "drug_col_target_name", "zip_synergy_label", "split"]
    df = pd.read_csv(data, usecols=use)
    df["y"] = df["zip_synergy_label"].map(C2I).astype(np.int8)
    # context numerics (mirror prep.py)
    nt = lambda s: 0 if not isinstance(s, str) or s == "" else s.count(";") + 1
    df["n_targets_row"] = df.drug_row_target_name.map(nt).astype(np.int16)
    df["n_targets_col"] = df.drug_col_target_name.map(nt).astype(np.int16)
    df["has_target_row"] = (df.n_targets_row > 0).astype(np.int8)
    df["has_target_col"] = (df.n_targets_col > 0).astype(np.int8)
    ts = lambda s: frozenset() if not isinstance(s, str) or s == "" else frozenset(t.strip() for t in s.split(";"))
    df["shared_target"] = np.array([int(len(a & b) > 0)
                                    for a, b in zip(df.drug_row_target_name.map(ts),
                                                    df.drug_col_target_name.map(ts))], dtype=np.int8)
    for c in ["drug_row_clinical_phase", "drug_col_clinical_phase"]:
        df[c] = df[c].fillna(-1).astype(np.float32)

    sm = pd.read_csv(args.smiles)
    sm.columns = [c.lower() for c in sm.columns]
    smiles_map = dict(zip(sm["drug"], sm["smiles"]))
    drugs = pd.unique(pd.concat([df.drug_row, df.drug_col]))
    miss = [d for d in drugs if not isinstance(smiles_map.get(d), str) or not smiles_map.get(d)]
    if len(miss) > 0.5 * len(drugs):
        sys.exit(f"Only {len(drugs)-len(miss)}/{len(drugs)} drugs have SMILES; provide a fuller "
                 f"--smiles table (see fetch_drug_smiles.py). Aborting.")
    fps = build_fingerprints(drugs, smiles_map, args.nbits, args.radius)

    Xfp = pair_features(df, fps, args.nbits)
    cats = []
    if args.mode == "struct_ctx":
        ctx = df[CTX_CAT + CTX_NUM].copy()
        for c in CTX_CAT:
            ctx[c] = ctx[c].astype("category")
        cats = CTX_CAT
        X = pd.concat([Xfp, ctx], axis=1)
    else:
        X = Xfp

    tr, va, te = make_splits(df, args.regime)
    ytr, yva, yte = df.y.values[tr], df.y.values[va], df.y.values[te]
    cw = {i: len(ytr) / (3 * max(1, np.sum(ytr == i))) for i in range(3)}
    sw = np.array([cw[v] for v in ytr])
    g = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=600,
                           learning_rate=0.05, num_leaves=128, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                           reg_lambda=2.0, max_bin=255, force_col_wise=True,
                           random_state=RNG, n_jobs=-1)
    g.fit(X[tr], ytr, sample_weight=sw, eval_set=[(X[va], yva)], eval_metric="multi_logloss",
          categorical_feature=cats, callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
    proba = g.predict_proba(X[te]); pred = proba.argmax(1)

    res = {"model": f"LightGBM + Morgan ECFP (mode={args.mode})", "regime": args.regime,
           "nbits": args.nbits, "radius": args.radius,
           "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
           "n_drugs_missing_smiles": len(miss), "metrics": metrics_block(yte, pred, proba)}
    fn = out / f"fingerprint_{args.mode}_{args.regime}.json"
    json.dump(res, open(fn, "w"), indent=2)
    m = res["metrics"]
    print(f"[{args.mode}/{args.regime}] balacc={m['balanced_accuracy']:.4f} "
          f"mf1={m['macro_f1']:.4f} AUPRC(syn)={m['auprc']['synergistic']:.3f} "
          f"in {time.time()-t0:.1f}s -> {fn}")


if __name__ == "__main__":
    main()
