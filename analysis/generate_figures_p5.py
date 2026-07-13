#!/usr/bin/env python3
"""Generate Phase 5 figures for the analysis report."""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results/figures_p5", exist_ok=True)

# ── 1. Signal AUROC: Noise A vs Noise E ──
signals_aurocs = {
    "A (Unlearnable)": {"-loss_cv": 0.868, "token_loss_top20": 0.946, "-IFD": 0.830, "rho_score": 0.996},
    "E (Shortcut)":  {"loss_cv": 0.665, "token_loss_top20": 0.838, "-IFD": 0.901, "rho_score": 0.996},
}
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(4)
w = 0.35
a_vals = list(signals_aurocs["A (Unlearnable)"].values())
e_vals = list(signals_aurocs["E (Shortcut)"].values())
labels = list(signals_aurocs["A (Unlearnable)"].keys())
ax.bar(x - w/2, a_vals, w, label="Noise A (Unlearnable)", color="#FF4136", alpha=0.8)
ax.bar(x + w/2, e_vals, w, label="Noise E (Shortcut)", color="#0074D9", alpha=0.8)
for i, (a, e) in enumerate(zip(a_vals, e_vals)):
    ax.text(i - w/2, a + 0.02, f"{a:.3f}", ha="center", fontsize=8)
    ax.text(i + w/2, e + 0.02, f"{e:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("AUROC", fontsize=12); ax.set_ylim(0, 1.1)
ax.set_title("Phase 5: Signal Detection Power — Noise A vs Noise E", fontsize=14)
ax.legend(fontsize=10); ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3); ax.grid(True, alpha=0.3, axis="y")
fig.savefig("results/figures_p5/signal_auroc_comparison.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ── 2. Signal mean fingerprint ──
with open("results/signals_p5.json") as f: signals = json.load(f)
nt_order = ["clean", "unlearnable", "shortcut"]
nt_names = {"clean": "Clean", "unlearnable": "A: Unlearnable", "shortcut": "E: Shortcut"}
nt_colors = {"clean": "#2ECC40", "unlearnable": "#FF4136", "shortcut": "#0074D9"}

# Normalized signal means
sig_names = ["loss_mu", "loss_cv", "loss_trend", "token_loss_top20", "ifd", "rho_score"]
data_norm = np.zeros((len(nt_order), len(sig_names)))
data_raw = np.zeros((len(nt_order), len(sig_names)))
for i, nt in enumerate(nt_order):
    subset = [s for s in signals if s["noise_type"] == nt]
    for j, sn in enumerate(sig_names):
        vals = [s.get(sn, 0) for s in subset]
        data_raw[i, j] = np.mean(vals)
for j in range(len(sig_names)):
    col = data_raw[:, j]; vmin, vmax = col.min(), col.max()
    if vmax - vmin > 1e-8: data_norm[:, j] = (col - vmin) / (vmax - vmin)

fig, ax = plt.subplots(figsize=(10, 3))
im = ax.imshow(data_norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(sig_names))); ax.set_xticklabels(sig_names, fontsize=11)
ax.set_yticks(range(len(nt_order))); ax.set_yticklabels([nt_names[nt] for nt in nt_order], fontsize=11)
for i in range(len(nt_order)):
    for j in range(len(sig_names)):
        ax.text(j, i, f"{data_raw[i,j]:.3f}", ha="center", va="center", fontsize=9,
                color="white" if data_norm[i,j] < 0.3 or data_norm[i,j] > 0.7 else "black")
ax.set_title("Phase 5: Signal Heatmap — Normalized Means by Noise Type", fontsize=13)
plt.tight_layout()
fig.savefig("results/figures_p5/signal_heatmap_p5.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ── 3. Noise distribution per filter strategy ──
filter_results = {
    "full":       {"A": 0, "E": 0, "Clean": 0},
    "p1_filtered":{"A": 422, "E": 151, "Clean": 807},
    "random_drop":{"A": 55, "E": 113, "Clean": 1212},
}
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(3); w = 0.22
for j, (name, data) in enumerate(filter_results.items()):
    vals = [data["A"], data["E"], data["Clean"]]
    ax.bar(x + (j-1)*w, vals, w, label=name, color=["#FF4136","#0074D9","#2ECC40"][j], alpha=0.8)
    for i, v in enumerate(vals):
        if v > 0: ax.text(x[i] + (j-1)*w, v + 10, str(v), ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(["A (Unlearnable)", "E (Shortcut)", "Clean (FP)"], fontsize=10)
ax.set_ylabel("Samples Dropped", fontsize=12)
ax.set_title("Phase 5: Samples Dropped by Each Filter Strategy (1380 total)", fontsize=13)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
fig.savefig("results/figures_p5/filter_distribution.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ── 4. MT-Bench comparison ──
mt_bench = {"full": 7.59, "p1_filtered": 7.64, "random_drop": 7.81}
fig, ax = plt.subplots(figsize=(5, 3.5))
labels = list(mt_bench.keys()); scores = list(mt_bench.values())
colors = ["#888888", "#2E86AB", "#2ECC40"]
bars = ax.bar(range(3), scores, color=colors, edgecolor="white")
ax.axhline(scores[0], color="#888", linestyle="--", alpha=0.5, label=f"Full ({scores[0]:.2f})")
for bar, s in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{s:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(["Full", "P1-filtered", "Random-drop"], fontsize=11)
ax.set_ylabel("MT-Bench Score", fontsize=12); ax.set_ylim(7.4, 8.0)
ax.set_title("Phase 5: MT-Bench Scores", fontsize=13); ax.legend()
fig.savefig("results/figures_p5/mt_bench_p5.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ── 5. Three-experiment cross-comparison ──
experiments = ["Phase 4\n1.5B", "Phase 4\n3B", "Phase 5\n3B"]
p1_delta = [0.06, 0.05, 0.05]
random_delta = [0.48, 0.39, 0.22]
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(3); w = 0.3
ax.bar(x - w/2, p1_delta, w, label="P1-filtered vs Full", color="#2E86AB")
ax.bar(x + w/2, random_delta, w, label="Random-drop vs Full", color="#2ECC40")
for i, (p, r) in enumerate(zip(p1_delta, random_delta)):
    ax.text(i - w/2, p + 0.01, f"+{p:.2f}", ha="center", fontsize=9)
    ax.text(i + w/2, r + 0.01, f"+{r:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(experiments, fontsize=10)
ax.set_ylabel("MT-Bench Improvement over Full", fontsize=12)
ax.set_title("Cross-Experiment: Filtering Gains Are Consistent", fontsize=13)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
fig.savefig("results/figures_p5/cross_experiment_delta.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ── 6. ROC curves for Noise A and Noise E (both on same plot) ──
from sklearn.metrics import roc_curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, (noise_nt, noise_name) in enumerate([("unlearnable", "A: Unlearnable"), ("shortcut", "E: Shortcut")]):
    clean = [s for s in signals if s["noise_type"] == "clean"]
    noisy = [s for s in signals if s["noise_type"] == noise_nt]
    target = clean[:len(noisy)] + noisy
    y = np.array([0]*len(clean[:len(noisy)]) + [1]*len(noisy))
    feats = {
        "-loss_cv": np.array([-s["loss_cv"] for s in target]),
        "-loss_trend": np.array([-s["loss_trend"] for s in target]),
        "token_loss_top20": np.array([s.get("token_loss_top20",0) for s in target]),
        "-IFD": np.array([-s.get("ifd",0) for s in target]),
    }
    for name, scores in feats.items():
        fpr, tpr, _ = roc_curve(y, scores)
        axes[ax_idx].plot(fpr, tpr, linewidth=1.5, label=name)
    axes[ax_idx].plot([0,1], [0,1], "k--", alpha=0.3)
    axes[ax_idx].set_xlabel("FPR"); axes[ax_idx].set_ylabel("TPR")
    axes[ax_idx].set_title(f"ROC: {noise_name} vs Clean")
    axes[ax_idx].legend(fontsize=8); axes[ax_idx].grid(True, alpha=0.3)

fig.suptitle("Phase 5: ROC Curves — Noise A vs Noise E (1.5B)", fontsize=14)
plt.tight_layout()
fig.savefig("results/figures_p5/roc_noise_a_vs_e.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print("Phase 5 figures generated:")
for f in sorted(os.listdir("results/figures_p5")): print(f"  results/figures_p5/{f}")
