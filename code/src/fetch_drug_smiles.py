"""
Build a drug -> SMILES table for fingerprint_baseline.py.

Two sources, in order of preference:

1) DrugComb drug annotation (recommended, authoritative for this dataset):
   Download the drug table from the DrugComb portal/API (it includes SMILES) and pass it
   with --drugcomb path/to/drugcomb_drugs.csv. The script will match on drug name and use
   its SMILES/canonical_smiles column directly.

2) PubChem PUG REST by compound name (fallback; needs internet, rate-limited):
   With no --drugcomb file, the script queries PubChem for each unique drug name.

Usage:
  python fetch_drug_smiles.py --out drug_smiles.csv
  python fetch_drug_smiles.py --drugcomb drugcomb_drugs.csv --out drug_smiles.csv

Output: CSV with columns drug,smiles (drugs that could not be resolved get an empty cell).
"""
import argparse, os, sys, time, csv
from pathlib import Path
import pandas as pd


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
    sys.exit("Could not locate the modeling CSV; pass --data.")


def unique_drugs(data):
    df = pd.read_csv(data, usecols=["drug_row", "drug_col"])
    return sorted(set(df.drug_row) | set(df.drug_col))


def from_drugcomb(path, drugs):
    t = pd.read_csv(path)
    cols = {c.lower(): c for c in t.columns}
    name_col = next((cols[k] for k in ("dname", "drug_name", "name", "drug") if k in cols), None)
    smi_col = next((cols[k] for k in ("smiles", "canonical_smiles", "isomeric_smiles") if k in cols), None)
    if not name_col or not smi_col:
        sys.exit(f"--drugcomb file needs a name and a SMILES column; found {list(t.columns)}")
    m = {str(n).strip().lower(): s for n, s in zip(t[name_col], t[smi_col])}
    return {d: m.get(str(d).strip().lower(), "") for d in drugs}


def from_pubchem(drugs, pause=0.25):
    import urllib.parse, urllib.request, json
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"
    out = {}
    for i, d in enumerate(drugs, 1):
        smi = ""
        try:
            url = base.format(urllib.parse.quote(str(d)))
            with urllib.request.urlopen(url, timeout=20) as r:
                js = json.load(r)
            smi = js["PropertyTable"]["Properties"][0].get("CanonicalSMILES", "")
        except Exception:
            smi = ""
        out[d] = smi
        if i % 50 == 0:
            print(f"  {i}/{len(drugs)} resolved", flush=True)
        time.sleep(pause)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--drugcomb", default=None, help="DrugComb drug annotation CSV with SMILES")
    ap.add_argument("--out", default="drug_smiles.csv")
    args = ap.parse_args()
    drugs = unique_drugs(args.data or discover_data())
    print(f"{len(drugs)} unique drugs")
    mapping = from_drugcomb(args.drugcomb, drugs) if args.drugcomb else from_pubchem(drugs)
    n_ok = sum(1 for v in mapping.values() if isinstance(v, str) and v)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["drug", "smiles"])
        for d in drugs:
            w.writerow([d, mapping.get(d, "")])
    print(f"wrote {args.out}: {n_ok}/{len(drugs)} drugs with SMILES")


if __name__ == "__main__":
    main()
