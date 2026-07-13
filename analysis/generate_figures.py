#!/usr/bin/env python3
"""Generate additional figures for the analysis report."""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results/signals.json") as f:
    signals = json.load(f)

NOISE_COLORS = {
    "clean": "#2ECC40", "unlearnable": "#FF4136",
    "label_noise": "#FF851B", "redundant": "#B10DC9", "pseudo_quality": "#0074D9",
}
NOISE_NAMES = {"clean": "Clean", "unlearnable": "A: Unlearnable",
               "label_noise": "B: Label Noise", "redundant": "C: Redundant",
               "pseudo_quality": "D: Pseudo-Quality"}
NOISE_ORDER = ["clean", "unlearnable", "label_noise", "redundant", "pseudo_quality"]

os.makedirs("results/figures", exist_ok=True)

# ── 1. token_loss_top20 distribution histogram ──
fig, ax = plt.subplots(figsize=(10,5))
for nt in NOISE_ORDER:
    vals = [s["token_loss_top20"] for s in signals if s["noise_type"] == nt and "token_loss_top20" in s]
    ax.hist(vals, bins=40, alpha=0.5, color=NOISE_COLORS[nt], label=NOISE_NAMES[nt], density=True)
ax.axvline(0.36, color="#FF4136", linestyle="--", alpha=0.7, label="A mean (0.36)")
ax.axvline(0.68, color="#2ECC40", linestyle="--", alpha=0.7, label="Clean mean (0.68)")
ax.set_xlabel("token_loss_top20", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("P0 Signal: Token-Loss Top-20% Distribution by Noise Type", fontsize=13)
ax.legend(fontsize=8, ncol=2)
fig.savefig("results/figures/token_top20_dist.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 2. Loss CV distribution ──
fig, ax = plt.subplots(figsize=(10,5))
for nt in NOISE_ORDER:
    vals = [s["loss_cv"] for s in signals if s["noise_type"] == nt]
    ax.hist(vals, bins=50, alpha=0.5, color=NOISE_COLORS[nt], label=NOISE_NAMES[nt], density=True)
ax.set_xlabel("loss_cv (σ/μ)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("P1 Signal: Loss CV Distribution by Noise Type", fontsize=13)
ax.legend(fontsize=8)
fig.savefig("results/figures/loss_cv_dist.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 3. 5-epoch loss trajectories (line plot) ──
fig, ax = plt.subplots(figsize=(10,6))
epochs = [1, 2, 3, 4, 5]
for nt in NOISE_ORDER:
    subset = [s for s in signals if s["noise_type"] == nt]
    if not subset:
        continue
    avg_losses = []
    for e in epochs:
        lv = [s.get(f"epochs_1-{e}_loss_mu", s.get("loss_mu", 0)) for s in subset]
        avg_losses.append(np.mean(lv))
    ax.plot(epochs, avg_losses, "o-", color=NOISE_COLORS[nt], label=NOISE_NAMES[nt], linewidth=2, markersize=6)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Mean Loss", fontsize=12)
ax.set_title("Loss Trajectories Across 5 Epochs by Noise Type", fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.savefig("results/figures/loss_trajectories_line.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 4. signal heatmap ──
signal_names = ["loss_mu", "loss_cv", "loss_trend", "token_loss_top20", "ifd", "rho_score"]
data_matrix = np.zeros((len(NOISE_ORDER), len(signal_names)))
for i, nt in enumerate(NOISE_ORDER):
    subset = [s for s in signals if s["noise_type"] == nt]
    for j, sn in enumerate(signal_names):
        data_matrix[i, j] = np.mean([s.get(sn, 0) for s in subset])

# Normalize per column for heatmap
data_norm = np.zeros_like(data_matrix)
for j in range(len(signal_names)):
    col = data_matrix[:, j]
    vmin, vmax = col.min(), col.max()
    if vmax - vmin > 1e-8:
        data_norm[:, j] = (col - vmin) / (vmax - vmin)

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
ax.set_title("Signal Heatmap: Normalized Mean Values by Noise Type", fontsize=13)
plt.tight_layout()
fig.savefig("results/figures/signal_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 5. composite score distribution ──
fig, ax = plt.subplots(figsize=(10,5))
for nt in NOISE_ORDER:
    vals = [s["composite_score"] for s in signals if s["noise_type"] == nt and "composite_score" in s]
    ax.hist(vals, bins=50, alpha=0.5, color=NOISE_COLORS[nt], label=NOISE_NAMES[nt], density=True)
ax.set_xlabel("Composite Score (higher = more noise-like)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Phase 4: Composite Score Distribution (fix: −loss_cv + −loss_trend)", fontsize=13)
ax.legend(fontsize=8)
fig.savefig("results/figures/composite_score_dist.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 6. ROC curves ──
from sklearn.metrics import roc_curve
fig, ax = plt.subplots(figsize=(8,6))
clean_samples = [s for s in signals if s["noise_type"] == "clean"]
a_samples = [s for s in signals if s["noise_type"] == "unlearnable"]
target = clean_samples + a_samples
y = np.array([0 if s["noise_type"] == "clean" else 1 for s in target])

features = {
    "-loss_cv": np.array([-s["loss_cv"] for s in target]),
    "-loss_trend": np.array([-s["loss_trend"] for s in target]),
    "token_loss_top20": np.array([s.get("token_loss_top20",0) for s in target]),
    "joint": None,
}
cv_n = (features["-loss_cv"] - features["-loss_cv"].min()) / (features["-loss_cv"].max() - features["-loss_cv"].min() + 1e-8)
tr_n = (features["-loss_trend"] - features["-loss_trend"].min()) / (features["-loss_trend"].max() - features["-loss_trend"].min() + 1e-8)
features["joint"] = cv_n + tr_n

for name, scores in features.items():
    fpr, tpr, _ = roc_curve(y, scores)
    ax.plot(fpr, tpr, linewidth=2, label=f"{name}")

ax.plot([0,1], [0,1], "k--", alpha=0.3, label="Random")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves: Unlearnable Noise Detection (Q1)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.savefig("results/figures/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 7. Per-noise ROC ──
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
    joint2 = cv_n2 + tr_n2
    fpr, tpr, _ = roc_curve(y2, joint2)
    ax.plot(fpr, tpr, linewidth=2, label=f"{NOISE_NAMES[noise_nt]}", linestyle=linestyles[idx])
ax.plot([0,1], [0,1], "k--", alpha=0.3)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves: Per-Noise-Type Detection (Q4)", fontsize=14)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.savefig("results/figures/per_noise_roc.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 8. Phase 4 bar chart ──
mt_bench = {"full": 6.83, "p1_filtered": 6.89, "rho_filtered": 7.31, "random_drop": 7.24, "ifd_only": 7.18}
labels = ["full", "p1_filtered", "rho_filtered", "random_drop", "ifd_only"]
scores = [mt_bench[k] for k in labels]
colors = ["#888888", "#2E86AB", "#F18F01", "#2ECC40", "#B10DC9"]
fig, ax = plt.subplots(figsize=(8,4))
bars = ax.bar(range(len(labels)), scores, color=colors, edgecolor="white")
ax.axhline(scores[0], color="#888888", linestyle="--", alpha=0.5, label=f"Full baseline ({scores[0]})")
for bar, s in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{s:.2f}", ha="center", fontsize=10)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(["Full", "P1-filtered", "RHO-filtered", "Random-drop", "IFD-only"], fontsize=10)
ax.set_ylabel("MT-Bench Score", fontsize=12)
ax.set_title("Phase 4: MT-Bench Scores by Filtering Strategy", fontsize=13)
ax.legend()
fig.savefig("results/figures/mt_bench_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("All figures generated:")
for f in sorted(os.listdir("results/figures")):
    print(f"  results/figures/{f}")
