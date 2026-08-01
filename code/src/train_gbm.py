"""Train LightGBM; save preds + metrics + importance."""
import os
import json, time, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score, cohen_kappa_score)
import lightgbm as lgb
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
m_tr=(df.split=="train").values; m_va=(df.split=="validation").values; m_te=(df.split=="test").values
ytr,yva,yte=df.y.values[m_tr],df.y.values[m_va],df.y.values[m_te]
gbm_feats=CAT_COLS+NUM_PASSTHROUGH+EXTRA_NUM
Xg=df[gbm_feats].copy()
for c in CAT_COLS: Xg[c]=Xg[c].astype("category")
Xg_tr,Xg_va,Xg_te=Xg[m_tr],Xg[m_va],Xg[m_te]
cw={i:len(ytr)/(3*np.sum(ytr==i)) for i in range(3)}; sw=np.array([cw[v] for v in ytr])
print("Training LightGBM...",flush=True)
gbm=lgb.LGBMClassifier(objective="multiclass",num_class=3,n_estimators=350,
    learning_rate=0.1,num_leaves=64,min_child_samples=100,subsample=0.8,subsample_freq=1,
    colsample_bytree=0.8,reg_lambda=2.0,max_bin=255,force_col_wise=True,random_state=RNG,n_jobs=-1)
gbm.fit(Xg_tr,ytr,sample_weight=sw,eval_set=[(Xg_va,yva)],eval_metric="multi_logloss",
    categorical_feature=CAT_COLS,callbacks=[lgb.early_stopping(30),lgb.log_evaluation(0)])
gbm.booster_.save_model(str(OUT/"model_lightgbm.txt"))  # persist immediately
pred,proba=gbm.predict(Xg_te),gbm.predict_proba(Xg_te)
print(f"GBM trained best_iter={gbm.best_iteration_} at {time.time()-t0:.1f}s",flush=True)
def ev(ytrue,ypred,yproba):
    rep=classification_report(ytrue,ypred,target_names=CLASSES,output_dict=True,zero_division=0)
    auc=roc_auc_score(ytrue,yproba,multi_class="ovr",average="macro")
    return {"model":"LightGBM","accuracy":accuracy_score(ytrue,ypred),
        "balanced_accuracy":balanced_accuracy_score(ytrue,ypred),"macro_f1":f1_score(ytrue,ypred,average="macro"),
        "weighted_f1":f1_score(ytrue,ypred,average="weighted"),"cohen_kappa":cohen_kappa_score(ytrue,ypred),
        "macro_auc_ovr":auc,"per_class":{c:{"precision":rep[c]["precision"],"recall":rep[c]["recall"],
        "f1":rep[c]["f1-score"],"support":rep[c]["support"]} for c in CLASSES},
        "confusion_matrix":confusion_matrix(ytrue,ypred).tolist()}
d=ev(yte,pred,proba); d["best_iteration"]=int(gbm.best_iteration_ or gbm.n_estimators)
imp=pd.DataFrame({"feature":gbm_feats,"gain":gbm.booster_.feature_importance(importance_type="gain")}).sort_values("gain",ascending=False)
d["feature_importance"]=imp.to_dict(orient="records"); d["features_gbm"]=gbm_feats
maj=np.bincount(ytr).argmax()
d["majority_baseline"]={"accuracy":float((yte==maj).mean()),"macro_f1":float(f1_score(yte,np.full_like(yte,maj),average="macro",zero_division=0))}
d["dataset"]={"n_total":int(len(df)),"n_train":int(m_tr.sum()),"n_val":int(m_va.sum()),"n_test":int(m_te.sum()),
    "classes":CLASSES,"class_counts_total":{c:int((df.y==i).sum()) for i,c in enumerate(CLASSES)},
    "class_counts_train":{c:int((df[m_tr].y==i).sum()) for i,c in enumerate(CLASSES)},
    "n_studies":int(df.study_name.nunique()),"n_tissues":int(df.tissue_name.nunique()),
    "n_cell_lines":int(df.cell_line_name.nunique()),"n_drugs":int(pd.concat([df.drug_row,df.drug_col]).nunique())}
json.dump(d,open(OUT/"metrics_gbm.json","w"),indent=2)
imp.to_csv(OUT/"feature_importance.csv",index=False)
gbm.booster_.save_model(str(OUT/"model_lightgbm.txt"))
print(f"GBM acc={d['accuracy']:.4f} macroF1={d['macro_f1']:.4f} AUC={d['macro_auc_ovr']:.4f} bestiter={d['best_iteration']} in {time.time()-t0:.1f}s")
