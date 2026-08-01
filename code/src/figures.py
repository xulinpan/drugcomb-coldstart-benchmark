"""Generate all paper figures."""
import os
import json, warnings
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.preprocessing import label_binarize
import lightgbm as lgb
warnings.filterwarnings("ignore")

OUT = Path(os.environ.get("DRUGCOMB_OUT", "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
CLASSES = ["antagonistic", "additive", "synergistic"]
NICE = ["Antagonistic", "Additive", "Synergistic"]
CAT_COLS = ["study_name","tissue_name","cell_line_name","drug_row","drug_col"]
NUM_PASSTHROUGH = ["drug_row_clinical_phase","drug_col_clinical_phase"]
EXTRA_NUM = ["n_targets_row","n_targets_col","has_target_row","has_target_col","shared_target"]
BLUE, ORANGE, GREEN = "#2c6fbb", "#e08214", "#2ca25f"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

mlr = json.load(open(OUT/"metrics_lr.json"))
mg  = json.load(open(OUT/"metrics_gbm.json"))
lrp = np.load(OUT/"lr_preds.npz")
y_true = lrp["y_true"]; lr_proba = lrp["lr_proba"]

# recompute gbm proba from saved booster
df = pd.read_parquet(OUT/"processed.parquet")
m_te = (df.split=="test").values
Xg = df[CAT_COLS+NUM_PASSTHROUGH+EXTRA_NUM].copy()
for c in CAT_COLS: Xg[c]=Xg[c].astype("category")
booster = lgb.Booster(model_file=str(OUT/"model_lightgbm.txt"))
gbm_proba = booster.predict(Xg[m_te])
gbm_pred = gbm_proba.argmax(1)

# ---------- Fig 1: class distribution ----------
cc = mg["dataset"]["class_counts_total"]
fig, ax = plt.subplots(figsize=(6,4))
vals=[cc[c] for c in CLASSES]
bars=ax.bar(NICE, vals, color=[ORANGE,BLUE,GREEN])
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, v+5000, f"{v:,}\n({v/sum(vals)*100:.1f}%)", ha="center", fontsize=9)
ax.set_ylabel("Number of drug-pair combinations"); ax.set_title("ZIP synergy class distribution (n=739,964)")
ax.set_ylim(0, max(vals)*1.18)
fig.savefig(FIG/"fig1_class_distribution.png"); plt.close(fig)

# ---------- Fig 2: confusion matrices ----------
def cm_plot(ax, cm, title):
    cm=np.array(cm); cmn=cm/cm.sum(1,keepdims=True)
    im=ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(NICE, rotation=20, ha="right"); ax.set_yticklabels(NICE)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f"{cmn[i,j]*100:.1f}%\n{cm[i,j]:,}",ha="center",va="center",
                    color="white" if cmn[i,j]>0.5 else "black", fontsize=8)
    return im
fig,axes=plt.subplots(1,2,figsize=(12,5))
cm_plot(axes[0], mlr["confusion_matrix"], "Logistic Regression")
im=cm_plot(axes[1], mg["confusion_matrix"], "LightGBM")
fig.colorbar(im, ax=axes, fraction=0.025, label="Row-normalized rate")
fig.suptitle("Test-set confusion matrices", y=1.02)
fig.savefig(FIG/"fig2_confusion_matrices.png"); plt.close(fig)

# ---------- Fig 3: model comparison ----------
metrics=["accuracy","balanced_accuracy","macro_f1","weighted_f1","cohen_kappa","macro_auc_ovr"]
labels=["Accuracy","Balanced\nacc.","Macro-F1","Weighted\nF1","Cohen's\nκ","Macro-AUC"]
lr_v=[mlr[m] for m in metrics]; gb_v=[mg[m] for m in metrics]
x=np.arange(len(metrics)); w=0.38
fig,ax=plt.subplots(figsize=(9,4.5))
b1=ax.bar(x-w/2, lr_v, w, label="Logistic Regression", color=BLUE)
b2=ax.bar(x+w/2, gb_v, w, label="LightGBM", color=GREEN)
maj=mg["majority_baseline"]["accuracy"]
ax.axhline(maj, ls="--", color="gray", lw=1, label=f"Majority baseline ({maj:.2f})")
for bars in (b1,b2):
    for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0,1.02); ax.set_ylabel("Score")
ax.set_title("Test-set performance comparison"); ax.legend(loc="upper right", fontsize=9)
fig.savefig(FIG/"fig3_model_comparison.png"); plt.close(fig)

# ---------- Fig 4: feature importance ----------
imp=pd.read_csv(OUT/"feature_importance.csv").sort_values("gain")
pretty={"drug_row":"Drug A identity","drug_col":"Drug B identity","cell_line_name":"Cell line",
        "study_name":"Study","tissue_name":"Tissue","n_targets_row":"# targets (A)",
        "n_targets_col":"# targets (B)","shared_target":"Shared target","has_target_row":"Has target (A)",
        "has_target_col":"Has target (B)","drug_row_clinical_phase":"Clinical phase (A)",
        "drug_col_clinical_phase":"Clinical phase (B)"}
fig,ax=plt.subplots(figsize=(7,5))
ax.barh([pretty.get(f,f) for f in imp.feature], imp.gain, color=GREEN)
ax.set_xlabel("Total gain"); ax.set_title("LightGBM feature importance (gain)")
fig.savefig(FIG/"fig4_feature_importance.png"); plt.close(fig)

# ---------- Fig 5: ROC curves (one-vs-rest) ----------
yb=label_binarize(y_true, classes=[0,1,2])
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,proba,title in [(axes[0],lr_proba,"Logistic Regression"),(axes[1],gbm_proba,"LightGBM")]:
    for i,(c,col) in enumerate(zip(NICE,[ORANGE,BLUE,GREEN])):
        fpr,tpr,_=roc_curve(yb[:,i], proba[:,i]); a=sk_auc(fpr,tpr)
        ax.plot(fpr,tpr,color=col,lw=2,label=f"{c} (AUC={a:.3f})")
    ax.plot([0,1],[0,1],"--",color="gray",lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=9)
fig.suptitle("One-vs-rest ROC curves (test set)", y=1.02)
fig.savefig(FIG/"fig5_roc_curves.png"); plt.close(fig)

# ---------- Fig 6: per-class F1 ----------
fig,ax=plt.subplots(figsize=(7,4.5))
lr_f1=[mlr["per_class"][c]["f1"] for c in CLASSES]
gb_f1=[mg["per_class"][c]["f1"] for c in CLASSES]
x=np.arange(3)
b1=ax.bar(x-w/2, lr_f1, w, label="Logistic Regression", color=BLUE)
b2=ax.bar(x+w/2, gb_f1, w, label="LightGBM", color=GREEN)
for bars in (b1,b2):
    for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f"{b.get_height():.2f}",ha="center",fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(NICE); ax.set_ylim(0,1.0); ax.set_ylabel("F1 score")
ax.set_title("Per-class F1 (test set)"); ax.legend()
fig.savefig(FIG/"fig6_per_class_f1.png"); plt.close(fig)

print("Figures written:", sorted(p.name for p in FIG.glob("*.png")))
print(f"GBM recomputed: acc={(gbm_pred==y_true).mean():.4f}")
