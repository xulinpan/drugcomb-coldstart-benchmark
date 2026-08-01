"""Train logistic baseline; save preds + metrics."""
import os
import json, time, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score, cohen_kappa_score)
import joblib
warnings.filterwarnings("ignore")
RNG=42
OUT=Path(os.environ.get("DRUGCOMB_OUT", "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
CLASSES=["antagonistic","additive","synergistic"]
CAT_COLS=["study_name","tissue_name","cell_line_name","drug_row","drug_col"]
NUM_PASSTHROUGH=["drug_row_clinical_phase","drug_col_clinical_phase"]
EXTRA_NUM=["n_targets_row","n_targets_col","has_target_row","has_target_col","shared_target"]
t0=time.time()
df=pd.read_parquet(OUT/"processed.parquet")
m_tr=(df.split=="train").values; m_te=(df.split=="test").values
ytr,yte=df.y.values[m_tr],df.y.values[m_te]
for i,c in enumerate(CLASSES): df[f"is_{c}"]=(df.y==i).astype(np.int8)
tr=df[m_tr]
def freq_encode(col):
    fm=tr[col].value_counts(normalize=True); return df[col].map(fm).fillna(0.0).values.astype(np.float32)
def target_encode(col,oh,s=20.0):
    g=tr[oh].mean(); a=tr.groupby(col)[oh].agg(["mean","count"])
    e=(a["mean"]*a["count"]+g*s)/(a["count"]+s); return df[col].map(e).fillna(g).values.astype(np.float32)
feat_lr,X=[],{}
for col in CAT_COLS:
    X[f"freq_{col}"]=freq_encode(col); feat_lr.append(f"freq_{col}")
    for c in CLASSES:
        nm=f"te_{col}_{c}"; X[nm]=target_encode(col,f"is_{c}"); feat_lr.append(nm)
for col in NUM_PASSTHROUGH+EXTRA_NUM:
    X[col]=df[col].values.astype(np.float32); feat_lr.append(col)
Xdf=pd.DataFrame(X); Xtr,Xte=Xdf.values[m_tr],Xdf.values[m_te]
sc=StandardScaler().fit(Xtr)
lr=LogisticRegression(max_iter=1000,C=1.0,class_weight="balanced",multi_class="multinomial",n_jobs=-1,random_state=RNG)
lr.fit(sc.transform(Xtr),ytr)
pred=lr.predict(sc.transform(Xte)); proba=lr.predict_proba(sc.transform(Xte))
def ev(ytrue,ypred,yproba):
    rep=classification_report(ytrue,ypred,target_names=CLASSES,output_dict=True,zero_division=0)
    auc=roc_auc_score(ytrue,yproba,multi_class="ovr",average="macro")
    return {"model":"Logistic Regression","accuracy":accuracy_score(ytrue,ypred),
        "balanced_accuracy":balanced_accuracy_score(ytrue,ypred),"macro_f1":f1_score(ytrue,ypred,average="macro"),
        "weighted_f1":f1_score(ytrue,ypred,average="weighted"),"cohen_kappa":cohen_kappa_score(ytrue,ypred),
        "macro_auc_ovr":auc,"per_class":{c:{"precision":rep[c]["precision"],"recall":rep[c]["recall"],
        "f1":rep[c]["f1-score"],"support":rep[c]["support"]} for c in CLASSES},
        "confusion_matrix":confusion_matrix(ytrue,ypred).tolist()}
d=ev(yte,pred,proba); d["features_lr"]=feat_lr
json.dump(d,open(OUT/"metrics_lr.json","w"),indent=2)
np.savez(OUT/"lr_preds.npz",y_true=yte,lr_pred=pred,lr_proba=proba)
joblib.dump({"model":lr,"scaler":sc,"features":feat_lr},OUT/"model_logreg.joblib")
print(f"LR acc={d['accuracy']:.4f} macroF1={d['macro_f1']:.4f} AUC={d['macro_auc_ovr']:.4f} in {time.time()-t0:.1f}s")
