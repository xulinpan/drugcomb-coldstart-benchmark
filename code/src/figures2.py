"""Figures for the revised paper (review-driven analyses)."""
import os
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUT=Path(os.environ.get("DRUGCOMB_OUT", "outputs"))
OUT.mkdir(parents=True, exist_ok=True); FIG=OUT/"figures"
CLASSES=["antagonistic","additive","synergistic"]; NICE=["Antagonistic","Additive","Synergistic"]
BLUE,ORANGE,GREEN,RED="#2c6fbb","#e08214","#2ca25f","#c0392b"
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":150,"savefig.bbox":"tight"})
pr=json.load(open(OUT/"pr.json")); pm=json.load(open(OUT/"perm.json"))
cd=json.load(open(OUT/"colddrug.json")); cc=json.load(open(OUT/"coldcell.json"))
tis=json.load(open(OUT/"tissue.json")); gbm=json.load(open(OUT/"metrics_gbm.json"))

# Fig7: PR curves
fig,axes=plt.subplots(1,3,figsize=(14,4.3))
for i,(c,nm) in enumerate(zip(CLASSES,NICE)):
    ax=axes[i]
    for tag,col,lab in [("curve_gbm",GREEN,"LightGBM"),("curve_lr",BLUE,"Logistic")]:
        cv=pr[tag][c]; ax.plot(cv["recall"],cv["precision"],color=col,lw=2,
            label=f"{lab} (AP={pr['auprc_gbm' if tag=='curve_gbm' else 'auprc_lr'][c]:.2f})")
    ax.axhline(pr["prior"][c],ls="--",color="gray",lw=1,label=f"Prior ({pr['prior'][c]:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(nm); ax.set_ylim(0,1.02)
    ax.legend(loc="upper right",fontsize=8)
fig.suptitle("Precision–recall curves by class (test set)",y=1.03)
fig.savefig(FIG/"fig7_pr_curves.png"); plt.close(fig)

# Fig8: permutation importance
pretty={"drug_row":"Drug A identity","drug_col":"Drug B identity","cell_line_name":"Cell line",
    "study_name":"Study","tissue_name":"Tissue","n_targets_row":"# targets (A)","n_targets_col":"# targets (B)",
    "shared_target":"Shared target","has_target_row":"Has target (A)","has_target_col":"Has target (B)",
    "drug_row_clinical_phase":"Clinical phase (A)","drug_col_clinical_phase":"Clinical phase (B)"}
items=sorted(pm["drops"].items(),key=lambda kv:kv[1])
fig,ax=plt.subplots(figsize=(7.5,5))
vals=[v for _,v in items]; names=[pretty.get(k,k) for k,_ in items]
ax.barh(names,vals,color=[GREEN if v>=0 else RED for v in vals])
ax.set_xlabel("Macro-F1 drop when feature permuted"); ax.axvline(0,color="black",lw=0.6)
ax.set_title("Permutation importance (LightGBM, test set)")
fig.savefig(FIG/"fig8_permutation_importance.png"); plt.close(fig)

# Fig9: cold-start comparison
scn=["Random\nsplit","Leave-drug-out","Leave-cell-out"]
balacc=[gbm["balanced_accuracy"],cd["metrics"]["balanced_accuracy"],cc["metrics"]["balanced_accuracy"]]
mf1=[gbm["macro_f1"],cd["metrics"]["macro_f1"],cc["metrics"]["macro_f1"]]
auc=[gbm["macro_auc_ovr"],cd["metrics"]["macro_auc_ovr"],cc["metrics"]["macro_auc_ovr"]]
x=np.arange(3); w=0.26
fig,ax=plt.subplots(figsize=(8,4.6))
for k,(vals,col,lab) in enumerate([(balacc,GREEN,"Balanced acc."),(mf1,BLUE,"Macro-F1"),(auc,ORANGE,"Macro-AUC")]):
    b=ax.bar(x+(k-1)*w,vals,w,color=col,label=lab)
    for r in b: ax.text(r.get_x()+r.get_width()/2,r.get_height()+0.01,f"{r.get_height():.2f}",ha="center",fontsize=8)
ax.axhline(1/3,ls=":",color="gray",lw=1,label="Random (bal.acc 0.33)")
ax.set_xticks(x); ax.set_xticklabels(scn); ax.set_ylim(0,1.0); ax.set_ylabel("Score")
ax.set_title("Generalization across evaluation regimes (LightGBM)"); ax.legend(fontsize=8,ncol=2,loc="upper right")
fig.savefig(FIG/"fig9_coldstart.png"); plt.close(fig)

# Fig10: per-tissue
ts=list(tis.items())[:12]
names=[t[0].replace("_"," ") for t in ts]; ba=[t[1]["balanced_accuracy"] for t in ts]; ns=[t[1]["n"] for t in ts]
fig,ax=plt.subplots(figsize=(8.5,5))
b=ax.barh(names[::-1],ba[::-1],color=GREEN)
for r,n in zip(b,ns[::-1]): ax.text(r.get_width()+0.005,r.get_y()+r.get_height()/2,f"{r.get_width():.2f} (n={n:,})",va="center",fontsize=8)
ax.set_xlim(0,1.05); ax.set_xlabel("Balanced accuracy"); ax.set_title("Per-tissue test performance (LightGBM, tissues with n≥300)")
fig.savefig(FIG/"fig10_per_tissue.png"); plt.close(fig)

# Fig11: calibration
fig,ax=plt.subplots(figsize=(5.6,5))
for c,nm,col in zip(CLASSES,NICE,[ORANGE,BLUE,GREEN]):
    cal=pr["calibration"][c]; ax.plot(cal["mean_pred"],cal["frac_pos"],"o-",color=col,lw=1.8,ms=4,label=nm)
ax.plot([0,1],[0,1],"--",color="gray",lw=1,label="Perfect")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title("Reliability diagram (LightGBM)"); ax.legend(fontsize=9)
fig.savefig(FIG/"fig11_calibration.png"); plt.close(fig)
print("wrote:",sorted(p.name for p in FIG.glob("fig7*")+ list(FIG.glob('fig8*'))+list(FIG.glob('fig9*'))+list(FIG.glob('fig1[01]*'))))
