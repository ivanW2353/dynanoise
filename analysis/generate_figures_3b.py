#!/usr/bin/env python3
"""Generate 3B-specific figures for the analysis report."""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results/signals_3b.json") as f:
    signals = json.load(f)

NOISE_COLORS = {
    "clean": "#2ECC40", "unlearnable": "#FF4136",
    "label_noise": "#FF851B", "redundant": "#B10DC9", "pseudo_quality": "#0074D9",
}
NOISE_NAMES = {"clean": "Clean", "unlearnable": "A: Unlearnable",
               "label_noise": "B: Label Noise", "redundant": "C: Redundant",
               "pseudo_quality": "D: Pseudo-Quality"}
NOISE_ORDER = ["clean", "unlearnable", "label_noise", "redundant", "pseudo_quality"]

os.makedirs("results/figures_3b", exist_ok=True)

# ── 1. ROC curves ──
from sklearn.metrics import roc_curve
fig, ax = plt.subplots(figsize=(8,6))
clean_samples = [s for s in signals if s["noise_type"] == "clean"]
a_samples = [s for s in signals if s["noise_type"] == "unlearnable"]
target = clean_samples + a_samples
y = np.array([0 if s["noise_type"] == "clean" else 1 for s in target])
features = {
    "-loss_cv (AUROC=0.901)": np.array([-s["loss_cv"] for s in target]),
    "-loss_trend (AUROC=0.573)": np.array([-s["loss_trend"] for s in target]),
    "token_loss_top20 (AUROC=0.947)": np.array([s.get("token_loss_top20",0) for s in target]),
}
for name, scores in features.items():
    fpr, tpr, _ = roc_curve(y, scores)
    ax.plot(fpr, tpr, linewidth=2, label=name)
ax.plot([0,1], [0,1], "k--", alpha=0.3, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves: Unlearnable Noise Detection (3B, Q1)", fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.savefig("results/figures_3b/roc_curves_3b.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 2. Signal heatmap ──
signal_names = ["loss_mu", "loss_cv", "loss_trend", "token_loss_top20", "ifd", "rho_score"]
data_matrix = np.zeros((len(NOISE_ORDER), len(signal_names)))
for i, nt in enumerate(NOISE_ORDER):
    subset = [s for s in signals if s["noise_type"] == nt]
    for j, sn in enumerate(signal_names):
        data_matrix[i, j] = np.mean([s.get(sn, 0) for s in subset])
data_norm = np.zeros_like(data_matrix)
for j in range(len(signal_names)):
    col = data_matrix[:, j]; vmin, vmax = col.min(), col.max()
    if vmax - vmin > 1e-8: data_norm[:, j] = (col - vmin) / (vmax - vmin)
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(data_norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(signal_names)))
ax.set_xticklabels(signal_names, fontsize=11, rotation=30, ha="right")
ax.set_yticks(range(len(NOISE_ORDER)))
ax.set_yticklabels([NOISE_NAMES[nt] for nt in NOISE_ORDER], fontsize=11)
for i in range(len(NOISE_ORDER)):
    for j in range(len(signal_names)):
        ax.text(j, i, f"{data_matrix[i,j]:.3f}", ha="center", va="center", fontsize=8,
                color="white" if data_norm[i,j] < 0.3 or data_norm[i,j] > 0.7 else "black")
ax.set_title("Signal Heatmap (3B): Normalized Mean Values by Noise Type", fontsize=13)
plt.tight_layout()
fig.savefig("results/figures_3b/signal_heatmap_3b.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 3. AUROC by epoch ──
with open("results/tables_3b/q3_results.json") as f: q3 = json.load(f)
fig, ax = plt.subplots(figsize=(8,5))
windows, auroc_cv, auroc_joint = [], [], []
for w in ["epochs_1-2","epochs_1-3","epochs_1-4","epochs_1-5"]:
    if w in q3: windows.append(q3[w]["n_epochs"]); auroc_cv.append(q3[w]["auroc_cv"]); auroc_joint.append(q3[w]["auroc_joint"])
ax.plot(windows, auroc_cv, "o-", color="#2E86AB", label="loss_cv only", linewidth=2)
ax.plot(windows, auroc_joint, "s-", color="#A23B72", label="joint (cv+trend)", linewidth=2)
ax.axhline(0.75, color="gray", linestyle="--", alpha=0.5, label="AUROC = 0.75")
ax.set_xlabel("Cumulative Epochs", fontsize=12); ax.set_ylabel("AUROC", fontsize=12)
ax.set_title("Q3: P1 Signal Quality vs Training Epochs (3B)", fontsize=14)
ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
fig.savefig("results/figures_3b/auroc_by_epoch_3b.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 4. Per-noise ROC ──
fig, ax = plt.subplots(figsize=(8,6))
noise_list = ["unlearnable", "label_noise", "redundant", "pseudo_quality"]
linestyles = ["-", "--", "-.", ":"]
for idx, noise_nt in enumerate(noise_list):
    n_samples = [s for s in signals if s["noise_type"] == noise_nt]
    target2 = clean_samples[:len(n_samples)] + n_samples
    y2 = np.array([0]*len(clean_samples[:len(n_samples)]) + [1]*len(n_samples))
    cv_vals = np.array([-s["loss_cv"] for s in target2])
    tr_vals = np.array([-s["loss_trend"] for s in target2])
    cv_n2 = (cv_vals - cv_vals.min()) / (cv_vals.max() - cv_vals.min() + 1e-8)
    tr_n2 = (tr_vals - tr_vals.min()) / (tr_vals.max() - tr_vals.min() + 1e-8)
    fpr, tpr, _ = roc_curve(y2, cv_n2 + tr_n2)
    ax.plot(fpr, tpr, linewidth=2, label=f"{NOISE_NAMES[noise_nt]}", linestyle=linestyles[idx])
ax.plot([0,1], [0,1], "k--", alpha=0.3)
ax.set_xlabel("False Positive Rate", fontsize=12); ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves: Per-Noise-Type Detection (3B, Q4)", fontsize=14)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
fig.savefig("results/figures_3b/per_noise_roc_3b.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 5. Cross-model comparison bar chart ──
metrics_1b = {"-loss_cv": 0.873, "joint": 0.961, "token_top20": 0.946, "P1+P0": 0.967}
metrics_3b = {"-loss_cv": 0.901, "joint": 0.850, "token_top20": 0.947, "P1+P0": 0.969}
labels = list(metrics_1b.keys())
x = np.arange(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(9,5))
ax.bar(x - w/2, [metrics_1b[k] for k in labels], w, label="1.5B", color="#2E86AB")
ax.bar(x + w/2, [metrics_3b[k] for k in labels], w, label="3B", color="#F18F01")
for i, k in enumerate(labels):
    ax.text(i-w/2, metrics_1b[k]+0.01, f'{metrics_1b[k]:.3f}', ha='center', fontsize=8)
    ax.text(i+w/2, metrics_3b[k]+0.01, f'{metrics_3b[k]:.3f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("AUROC", fontsize=12); ax.set_title("Cross-Model Signal Stability: 1.5B vs 3B", fontsize=13)
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
fig.savefig("results/figures_3b/cross_model_compare.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("3B Figures generated:")
for f in sorted(os.listdir("results/figures_3b")): print(f"  results/figures_3b/{f}")
