"""Regenerate the 7 manuscript figures at 300 DPI from the JSON metric files only
(no large dataset required). Outputs overwrite ../manuscript/figures/.
Run: python3 regen_figures_300dpi.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "manuscript" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DPI = 300

CLASSES = ["antagonistic", "additive", "synergistic"]
NICE = ["Antagonistic", "Additive", "Synergistic"]
BLUE, ORANGE, GREEN, RED = "#2c6fbb", "#e08214", "#2ca25f", "#c0392b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight"})

def J(name): return json.load(open(HERE / name))
mlr = J("metrics_lr.json"); mg = J("metrics_gbm.json")
pr = J("pr.json"); pm = J("perm.json")
cd = J("colddrug.json"); cc = J("coldcell.json"); tis = J("tissue.json")

# ---------- Fig 1: class distribution ----------
cdist = mg["dataset"]["class_counts_total"]; vals = [cdist[c] for c in CLASSES]
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(NICE, vals, color=[ORANGE, BLUE, GREEN])
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+5000, f"{v:,}\n({v/sum(vals)*100:.1f}%)", ha="center", fontsize=9)
ax.set_ylabel("Number of drug-pair combinations")
ax.set_title("ZIP synergy class distribution (n=739,964)")
ax.set_ylim(0, max(vals)*1.18)
fig.savefig(FIG/"fig1_class_distribution.png"); plt.close(fig)

# ---------- Fig 2: confusion matrices ----------
def cm_plot(ax, cm, title, show_y=True):
    cm = np.array(cm); cmn = cm/cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(NICE, rotation=20, ha="right")
    # Only the left panel carries y tick labels; otherwise the right panel's labels
    # collide with the left panel's matrix (reviewer 2, minor comment 2).
    if show_y:
        ax.set_yticklabels(NICE); ax.set_ylabel("True")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Predicted"); ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i,j]*100:.1f}%\n{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    return im
# NB: do not use sharey here - a shared y axis makes the two panels share one tick
# formatter, so blanking the right panel's labels would also blank the left panel's.
fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.10})
cm_plot(axes[0], mlr["confusion_matrix"], "Logistic Regression", show_y=True)
im = cm_plot(axes[1], mg["confusion_matrix"], "LightGBM", show_y=False)
fig.colorbar(im, ax=axes, fraction=0.025, label="Row-normalized rate")
fig.suptitle("Test-set confusion matrices", y=1.02)
fig.savefig(FIG/"fig2_confusion_matrices.png"); plt.close(fig)

# ---------- Fig 7: PR curves ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
for i, (c, nm) in enumerate(zip(CLASSES, NICE)):
    ax = axes[i]
    for tag, col, lab in [("curve_gbm", GREEN, "LightGBM"), ("curve_lr", BLUE, "Logistic")]:
        cv = pr[tag][c]
        key = "auprc_gbm" if tag == "curve_gbm" else "auprc_lr"
        ax.plot(cv["recall"], cv["precision"], color=col, lw=2, label=f"{lab} (AP={pr[key][c]:.2f})")
    ax.axhline(pr["prior"][c], ls="--", color="gray", lw=1, label=f"Prior ({pr['prior'][c]:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(nm); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=8)
fig.suptitle("Precision–recall curves by class (test set)", y=1.03)
fig.savefig(FIG/"fig7_pr_curves.png"); plt.close(fig)

# ---------- Fig 8: permutation importance ----------
pretty = {"drug_row": "Drug A identity", "drug_col": "Drug B identity", "cell_line_name": "Cell line",
          "study_name": "Study", "tissue_name": "Tissue", "n_targets_row": "# targets (A)",
          "n_targets_col": "# targets (B)", "shared_target": "Shared target",
          "has_target_row": "Has target (A)", "has_target_col": "Has target (B)",
          "drug_row_clinical_phase": "Clinical phase (A)", "drug_col_clinical_phase": "Clinical phase (B)"}
items = sorted(pm["drops"].items(), key=lambda kv: kv[1])
vals = [v for _, v in items]; names = [pretty.get(k, k) for k, _ in items]
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.barh(names, vals, color=[GREEN if v >= 0 else RED for v in vals])
ax.set_xlabel("Macro-F1 drop when feature permuted"); ax.axvline(0, color="black", lw=0.6)
ax.set_title("Permutation importance (LightGBM, test set)")
fig.savefig(FIG/"fig8_permutation_importance.png"); plt.close(fig)

# ---------- Fig 9: cold-start ----------
scn = ["Random\nsplit", "Leave-drug-out", "Leave-cell-out"]
balacc = [mg["balanced_accuracy"], cd["metrics"]["balanced_accuracy"], cc["metrics"]["balanced_accuracy"]]
mf1 = [mg["macro_f1"], cd["metrics"]["macro_f1"], cc["metrics"]["macro_f1"]]
auc = [mg["macro_auc_ovr"], cd["metrics"]["macro_auc_ovr"], cc["metrics"]["macro_auc_ovr"]]
x = np.arange(3); w = 0.26
fig, ax = plt.subplots(figsize=(8, 4.6))
for k, (vv, col, lab) in enumerate([(balacc, GREEN, "Balanced acc."), (mf1, BLUE, "Macro-F1"), (auc, ORANGE, "Macro-AUC")]):
    b = ax.bar(x+(k-1)*w, vv, w, color=col, label=lab)
    for r in b:
        ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.01, f"{r.get_height():.2f}", ha="center", fontsize=8)
ax.axhline(1/3, ls=":", color="gray", lw=1, label="Random (bal.acc 0.33)")
ax.set_xticks(x); ax.set_xticklabels(scn); ax.set_ylim(0, 1.0); ax.set_ylabel("Score")
ax.set_title("Generalization across evaluation regimes (LightGBM)")
ax.legend(fontsize=8, ncol=2, loc="upper right")
fig.savefig(FIG/"fig9_coldstart.png"); plt.close(fig)

# ---------- Fig 10: per-tissue ----------
ts = list(tis.items())[:12]
names = [t[0].replace("_", " ") for t in ts]
ba = [t[1]["balanced_accuracy"] for t in ts]; ns = [t[1]["n"] for t in ts]
fig, ax = plt.subplots(figsize=(8.5, 5))
b = ax.barh(names[::-1], ba[::-1], color=GREEN)
for r, n in zip(b, ns[::-1]):
    ax.text(r.get_width()+0.005, r.get_y()+r.get_height()/2, f"{r.get_width():.2f} (n={n:,})", va="center", fontsize=8)
ax.set_xlim(0, 1.05); ax.set_xlabel("Balanced accuracy")
ax.set_title("Per-tissue test performance (LightGBM, tissues with n≥300)")
fig.savefig(FIG/"fig10_per_tissue.png"); plt.close(fig)

# ---------- Fig 11: calibration, BOTH models (reviewer 2, comment 4) ----------
oc_path = HERE / "review_experiments_output" / "ordinal_calibration.json"
if oc_path.exists():
    oc = json.load(open(oc_path))
    panels = [("logistic", "Logistic Regression"), ("lightgbm", "LightGBM")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes, panels):
        cal = oc["calibration"][key]
        for c, nm, col in zip(CLASSES, NICE, [ORANGE, BLUE, GREEN]):
            r = cal["reliability"][c]
            ax.plot(r["mean_pred"], r["frac_pos"], "o-", color=col, lw=1.8, ms=4, label=nm)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect")
        ax.set_xlabel("Mean predicted probability")
        ax.set_title("%s (ECE %.3f, Brier %.3f)"
                     % (title, cal["top_label_ece"], cal["brier_multiclass"]))
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(fontsize=9)
    fig.suptitle("Reliability diagrams", y=1.02)
    fig.savefig(FIG/"fig11_calibration.png"); plt.close(fig)
else:  # fall back to the single-model version if the new analysis has not been run
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for c, nm, col in zip(CLASSES, NICE, [ORANGE, BLUE, GREEN]):
        cal = pr["calibration"][c]
        ax.plot(cal["mean_pred"], cal["frac_pos"], "o-", color=col, lw=1.8, ms=4, label=nm)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (LightGBM)"); ax.legend(fontsize=9)
    fig.savefig(FIG/"fig11_calibration.png"); plt.close(fig)

print(f"Regenerated 7 figures at {DPI} DPI in {FIG}")
