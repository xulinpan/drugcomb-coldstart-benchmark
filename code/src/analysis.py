"""Resumable extended analysis addressing peer-review points.
Usage: python analysis.py <stage> [<stage> ...]
Stages: boot  ci+McNemar on random split
        pr    PR curves / AUPRC + calibration data
        perm  permutation importance
        tissue per-tissue breakdown
        colddrug  leave-drug-out train+eval
        coldcell  leave-cell-out train+eval
Each writes JSON/NPZ; skips if output already exists.
"""
import os
import sys, json, time, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score, average_precision_score,
    precision_recall_curve)
import lightgbm as lgb
warnings.filterwarnings("ignore")
RNG=42
OUT=Path(os.environ.get("DRUGCOMB_OUT", "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
CLASSES=["antagonistic","additive","synergistic"]
CAT=["study_name","tissue_name","cell_line_name","drug_row","drug_col"]
NUMP=["drug_row_clinical_phase","drug_col_clinical_phase"]
EXTRA=["n_targets_row","n_targets_col","has_target_row","has_target_col","shared_target"]
GFEAT=CAT+NUMP+EXTRA

def load():
    return pd.read_parquet(OUT/"processed.parquet")

def metrics_block(yt,yp,yproba):
    from sklearn.metrics import classification_report
    rep=classification_report(yt,yp,target_names=CLASSES,output_dict=True,zero_division=0)
    return {"accuracy":float(accuracy_score(yt,yp)),
        "balanced_accuracy":float(balanced_accuracy_score(yt,yp)),
        "macro_f1":float(f1_score(yt,yp,average="macro")),
        "weighted_f1":float(f1_score(yt,yp,average="weighted")),
        "cohen_kappa":float(cohen_kappa_score(yt,yp)),
        "macro_auc_ovr":float(roc_auc_score(yt,yproba,multi_class="ovr",average="macro")),
        "per_class":{c:{"precision":rep[c]["precision"],"recall":rep[c]["recall"],
            "f1":rep[c]["f1-score"],"support":rep[c]["support"]} for c in CLASSES},
        "confusion_matrix":confusion_matrix(yt,yp).tolist()}

def train_gbm(df,m_tr,m_va,m_te,n_est=400):
    ytr,yva,yte=df.y.values[m_tr],df.y.values[m_va],df.y.values[m_te]
    Xg=df[GFEAT].copy()
    for c in CAT: Xg[c]=Xg[c].astype("category")
    cw={i:len(ytr)/(3*max(1,np.sum(ytr==i))) for i in range(3)}
    sw=np.array([cw[v] for v in ytr])
    g=lgb.LGBMClassifier(objective="multiclass",num_class=3,n_estimators=n_est,
        learning_rate=0.1,num_leaves=64,min_child_samples=100,subsample=0.8,
        subsample_freq=1,colsample_bytree=0.8,reg_lambda=2.0,max_bin=255,
        force_col_wise=True,random_state=RNG,n_jobs=-1)
    g.fit(Xg[m_tr],ytr,sample_weight=sw,eval_set=[(Xg[m_va],yva)],
        eval_metric="multi_logloss",categorical_feature=CAT,
        callbacks=[lgb.early_stopping(30),lgb.log_evaluation(0)])
    proba=g.predict_proba(Xg[m_te]); pred=proba.argmax(1)
    return g,pred,proba,yte

# ---------------- stage: boot ----------------
def stage_boot():
    out=OUT/"boot.json"
    if out.exists(): print("boot skip"); return
    df=load(); m_te=(df.split=="test").values
    Xg=df[GFEAT].copy()
    for c in CAT: Xg[c]=Xg[c].astype("category")
    booster=lgb.Booster(model_file=str(OUT/"model_lightgbm.txt"))
    gproba=booster.predict(Xg[m_te]); gpred=gproba.argmax(1)
    lrp=np.load(OUT/"lr_preds.npz"); yt=lrp["y_true"]; lpred=lrp["lr_pred"]; lproba=lrp["lr_proba"]
    rng=np.random.default_rng(RNG); n=len(yt); B=300
    def boot_metrics(yp,yproba):
        accs=[];bals=[];mf1=[];aucs=[];kap=[]
        for _ in range(B):
            idx=rng.integers(0,n,n)
            accs.append(accuracy_score(yt[idx],yp[idx]))
            bals.append(balanced_accuracy_score(yt[idx],yp[idx]))
            mf1.append(f1_score(yt[idx],yp[idx],average="macro"))
            kap.append(cohen_kappa_score(yt[idx],yp[idx]))
            try: aucs.append(roc_auc_score(yt[idx],yproba[idx],multi_class="ovr",average="macro"))
            except: pass
        ci=lambda a:[float(np.percentile(a,2.5)),float(np.percentile(a,97.5)),float(np.mean(a))]
        return {"accuracy":ci(accs),"balanced_accuracy":ci(bals),"macro_f1":ci(mf1),
                "cohen_kappa":ci(kap),"macro_auc_ovr":ci(aucs)}
    # McNemar on correctness
    gc=(gpred==yt); lc=(lpred==yt)
    b=int(np.sum(gc & ~lc)); c=int(np.sum(~gc & lc))
    from math import isclose
    stat=(abs(b-c)-1)**2/(b+c) if (b+c)>0 else 0.0
    # chi2 sf with df=1
    from scipy.stats import chi2
    p=float(chi2.sf(stat,1))
    res={"lightgbm":boot_metrics(gpred,gproba),"logreg":boot_metrics(lpred,lproba),
         "mcnemar":{"b_gbm_only_correct":b,"c_lr_only_correct":c,"chi2":float(stat),"p_value":p}}
    json.dump(res,open(out,"w"),indent=2); print("boot done")

# ---------------- stage: pr ----------------
def stage_pr():
    out=OUT/"pr.json"
    if out.exists(): print("pr skip"); return
    df=load(); m_te=(df.split=="test").values
    Xg=df[GFEAT].copy()
    for c in CAT: Xg[c]=Xg[c].astype("category")
    booster=lgb.Booster(model_file=str(OUT/"model_lightgbm.txt"))
    gproba=booster.predict(Xg[m_te]); yt=df.y.values[m_te]
    lrp=np.load(OUT/"lr_preds.npz"); lproba=lrp["lr_proba"]
    from sklearn.preprocessing import label_binarize
    yb=label_binarize(yt,classes=[0,1,2])
    res={"auprc_gbm":{},"auprc_lr":{},"prior":{},"curve_gbm":{},"curve_lr":{}}
    for i,c in enumerate(CLASSES):
        res["prior"][c]=float(yb[:,i].mean())
        res["auprc_gbm"][c]=float(average_precision_score(yb[:,i],gproba[:,i]))
        res["auprc_lr"][c]=float(average_precision_score(yb[:,i],lproba[:,i]))
        for tag,pr in [("curve_gbm",gproba),("curve_lr",lproba)]:
            p,r,_=precision_recall_curve(yb[:,i],pr[:,i])
            # subsample to 200 pts
            idx=np.linspace(0,len(p)-1,200).astype(int)
            res[tag][c]={"precision":p[idx].tolist(),"recall":r[idx].tolist()}
    # calibration for synergistic class (gbm)
    from sklearn.calibration import calibration_curve
    for i,c in enumerate(CLASSES):
        ft,mp=calibration_curve(yb[:,i],gproba[:,i],n_bins=10,strategy="quantile")
        res.setdefault("calibration",{})[c]={"frac_pos":ft.tolist(),"mean_pred":mp.tolist()}
    json.dump(res,open(out,"w"),indent=2); print("pr done")

# ---------------- stage: perm ----------------
def stage_perm():
    out=OUT/"perm.json"
    if out.exists(): print("perm skip"); return
    df=load(); m_te=(df.split=="test").values
    Xg=df[GFEAT].copy()
    for c in CAT: Xg[c]=Xg[c].astype("category")
    Xte=Xg[m_te].reset_index(drop=True); yt=df.y.values[m_te]
    # subsample 20000 for speed
    rng=np.random.default_rng(RNG)
    sub=rng.choice(len(Xte),min(20000,len(Xte)),replace=False)
    Xs=Xte.iloc[sub].reset_index(drop=True); ys=yt[sub]
    booster=lgb.Booster(model_file=str(OUT/"model_lightgbm.txt"))
    base=f1_score(ys,booster.predict(Xs).argmax(1),average="macro")
    res={"baseline_macro_f1":float(base),"drops":{}}
    for f in GFEAT:
        Xp=Xs.copy()
        Xp[f]=Xp[f].sample(frac=1.0,random_state=1).values
        if str(Xs[f].dtype)=="category": Xp[f]=Xp[f].astype(Xs[f].dtype)
        m=f1_score(ys,booster.predict(Xp).argmax(1),average="macro")
        res["drops"][f]=float(base-m)
    json.dump(res,open(out,"w"),indent=2); print("perm done")

# ---------------- stage: tissue ----------------
def stage_tissue():
    out=OUT/"tissue.json"
    if out.exists(): print("tissue skip"); return
    df=load(); m_te=(df.split=="test").values
    Xg=df[GFEAT].copy()
    for c in CAT: Xg[c]=Xg[c].astype("category")
    booster=lgb.Booster(model_file=str(OUT/"model_lightgbm.txt"))
    sub=df[m_te].copy(); sub["pred"]=booster.predict(Xg[m_te]).argmax(1)
    res={}
    for t,grp in sub.groupby("tissue_name"):
        if len(grp)<300: continue
        res[t]={"n":int(len(grp)),
            "balanced_accuracy":float(balanced_accuracy_score(grp.y,grp.pred)),
            "macro_f1":float(f1_score(grp.y,grp.pred,average="macro"))}
    res=dict(sorted(res.items(),key=lambda kv:-kv[1]["n"]))
    json.dump(res,open(out,"w"),indent=2); print("tissue done")

# ---------------- stage: colddrug ----------------
def stage_colddrug():
    out=OUT/"colddrug.json"
    if out.exists(): print("colddrug skip"); return
    df=load(); rng=np.random.default_rng(RNG)
    drugs=pd.unique(pd.concat([df.drug_row,df.drug_col]))
    unseen=set(rng.choice(drugs,int(len(drugs)*0.12),replace=False))
    inrow=df.drug_row.isin(unseen); incol=df.drug_col.isin(unseen)
    te=(inrow|incol).values            # at least one unseen drug
    pool=~te                            # both drugs seen -> train/val pool
    # carve val from pool
    pidx=np.where(pool)[0]; rng.shuffle(pidx)
    nval=int(len(pidx)*0.1); va=np.zeros(len(df),bool); va[pidx[:nval]]=True
    tr=pool.copy(); tr[pidx[:nval]]=False
    g,pred,proba,yte=train_gbm(df,tr,va,te)
    res={"split":"leave-drug-out (>=1 unseen drug in test)",
         "n_unseen_drugs":len(unseen),"n_train":int(tr.sum()),"n_val":int(va.sum()),
         "n_test":int(te.sum()),"metrics":metrics_block(yte,pred,proba)}
    json.dump(res,open(out,"w"),indent=2)
    print("colddrug done acc=%.4f balacc=%.4f mf1=%.4f"%(res["metrics"]["accuracy"],
        res["metrics"]["balanced_accuracy"],res["metrics"]["macro_f1"]))

# ---------------- stage: coldcell ----------------
def stage_coldcell():
    out=OUT/"coldcell.json"
    if out.exists(): print("coldcell skip"); return
    df=load(); rng=np.random.default_rng(RNG)
    cells=df.cell_line_name.unique()
    unseen=set(rng.choice(cells,int(len(cells)*0.15),replace=False))
    te=df.cell_line_name.isin(unseen).values
    pool=~te; pidx=np.where(pool)[0]; rng.shuffle(pidx)
    nval=int(len(pidx)*0.1); va=np.zeros(len(df),bool); va[pidx[:nval]]=True
    tr=pool.copy(); tr[pidx[:nval]]=False
    g,pred,proba,yte=train_gbm(df,tr,va,te)
    res={"split":"leave-cell-line-out","n_unseen_cells":len(unseen),
         "n_train":int(tr.sum()),"n_val":int(va.sum()),"n_test":int(te.sum()),
         "metrics":metrics_block(yte,pred,proba)}
    json.dump(res,open(out,"w"),indent=2)
    print("coldcell done acc=%.4f balacc=%.4f mf1=%.4f"%(res["metrics"]["accuracy"],
        res["metrics"]["balanced_accuracy"],res["metrics"]["macro_f1"]))

STAGES={"boot":stage_boot,"pr":stage_pr,"perm":stage_perm,"tissue":stage_tissue,
        "colddrug":stage_colddrug,"coldcell":stage_coldcell}
if __name__=="__main__":
    for s in sys.argv[1:]:
        t=time.time(); STAGES[s](); print(f"[{s}] {time.time()-t:.1f}s",flush=True)
