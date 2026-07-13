#!/usr/bin/env python3
"""
Visualization module for loss dynamics experiment.
Generates key plots used in the paper.
"""

import os
import json
import argparse
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_signals(path: str) -> list:
    with open(path) as f:
        return json.load(f)


NOISE_COLORS = {
    "clean": "#2ECC40",
    "unlearnable": "#FF4136",
    "label_noise": "#FF851B",
    "redundant": "#B10DC9",
    "pseudo_quality": "#0074D9",
}

NOISE_MARKERS = {
    "clean": "o",
    "unlearnable": "x",
    "label_noise": "s",
    "redundant": "^",
    "pseudo_quality": "d",
}


def plot_loss_cv_vs_trend(signals: list, output_dir: str):
    """Scatter plot: loss_cv x loss_trend, colored by noise_type."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for nt in sorted(NOISE_COLORS.keys()):
        subset = [s for s in signals if s["noise_type"] == nt]
        if not subset:
            continue
        x = [s.get("loss_cv", 0) for s in subset]
        y = [s.get("loss_trend", 0) for s in subset]
        ax.scatter(x, y, c=NOISE_COLORS[nt], label=nt, alpha=0.4, s=8, edgecolors="none")

    ax.set_xlabel("Loss CV (σ_L / μ_L)", fontsize=12)
    ax.set_ylabel("Loss Trend (linear slope)", fontsize=12)
    ax.set_title("P1 Signals: Loss Variability vs Training Trend", fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "loss_cv_vs_trend.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: loss_cv_vs_trend.png")


def plot_auroc_by_epoch(q3_results: dict, output_dir: str):
    """Line plot: AUROC by cumulative epoch window."""
    fig, ax = plt.subplots(figsize=(8, 5))

    windows = []
    auroc_cv = []
    auroc_joint = []

    for window in ["epochs_1-2", "epochs_1-3", "epochs_1-4", "epochs_1-5"]:
        if window in q3_results:
            windows.append(q3_results[window]["n_epochs"])
            auroc_cv.append(q3_results[window]["auroc_cv"])
            auroc_joint.append(q3_results[window]["auroc_joint"])

    if windows:
        ax.plot(windows, auroc_cv, "o-", color="#2E86AB", label="loss_cv only", linewidth=2)
        ax.plot(windows, auroc_joint, "s-", color="#A23B72", label="joint (cv + trend)", linewidth=2)
        ax.axhline(y=0.75, color="gray", linestyle="--", alpha=0.5, label="AUROC = 0.75")

        ax.set_xlabel("Cumulative Epochs", fontsize=12)
        ax.set_ylabel("AUROC", fontsize=12)
        ax.set_title("Q3: P1 Signal Quality vs Training Epochs", fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "auroc_by_epoch.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: auroc_by_epoch.png")


def plot_noise_type_auroc(q4_results: dict, output_dir: str):
    """Bar chart: AUROC for each noise type vs clean."""
    data = q4_results.get("per_noise_type", {})
    if not data:
        return

    noise_types = list(data.keys())
    auroc_joint = [data[nt]["auroc_joint"] for nt in noise_types]
    auroc_ifd = [data[nt]["auroc_ifd"] for nt in noise_types]

    x = np.arange(len(noise_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, auroc_joint, width, label="P1 Joint (cv+trend)", color="#2E86AB")
    bars2 = ax.bar(x + width / 2, auroc_ifd, width, label="IFD (baseline)", color="#F18F01")

    ax.set_xlabel("Noise Type", fontsize=12)
    ax.set_ylabel("AUROC vs Clean", fontsize=12)
    ax.set_title("Q4: Per-Noise-Type Discrimination (P1 vs IFD)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([nt.replace("_", "\n") for nt in noise_types], fontsize=9)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "noise_type_auroc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: noise_type_auroc.png")


def plot_loss_trajectories(signals: list, output_dir: str):
    """Line plot: loss trajectories for each noise type across epochs.
    Uses loss values from all_losses.json. Falls back to signal data."""
    fig, ax = plt.subplots(figsize=(10, 6))

    noise_types = ["clean", "unlearnable", "label_noise", "redundant", "pseudo_quality"]
    for nt in noise_types:
        subset = [s for s in signals if s["noise_type"] == nt]
        if not subset:
            continue

        loss_e1 = np.mean([s.get("epochs_1-2_loss_mu", s.get("loss_mu", 0)) for s in subset])
        loss_e5 = np.mean([s.get("epochs_1-5_loss_mu", s.get("loss_mu", 0)) for s in subset])

        cv_mean = np.mean([s.get("loss_cv", 0) for s in subset])

        ax.bar(
            nt.replace("_", "\n"),
            loss_e1,
            color=NOISE_COLORS.get(nt, "#999"),
            alpha=0.3,
            label=f"{nt} (epoch 1)" if nt == "clean" else "",
        )
        ax.bar(
            nt.replace("_", "\n"),
            loss_e5,
            color=NOISE_COLORS.get(nt, "#999"),
            alpha=0.7,
            label=f"{nt} (epoch 5)" if nt == "clean" else "",
        )

    ax.set_ylabel("Mean Loss", fontsize=12)
    ax.set_title("Loss Trajectories: Epoch 1 vs Epoch 5 by Noise Type", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "loss_trajectories.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: loss_trajectories.png")


def plot_radar_chart(q5_results: dict, output_dir: str):
    """Radar chart comparing 5 training groups on MT-Bench scores."""
    if not q5_results:
        return

    categories = list(q5_results.keys())
    scores = [q5_results[c] for c in categories]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    scores += scores[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, "o-", color="#2E86AB", linewidth=2)
    ax.fill(angles, scores, alpha=0.25, color="#2E86AB")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title("Downstream MT-Bench Scores by Training Group", fontsize=14, pad=20)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "radar_chart.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: radar_chart.png")


def main():
    parser = argparse.ArgumentParser(description="Visualization for experiment results")
    parser.add_argument("--signals-path", default="results/signals.json")
    parser.add_argument("--q3-results", default="results/tables/q3_results.json")
    parser.add_argument("--q4-results", default="results/tables/q4_results.json")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    signals = load_signals(args.signals_path)
    logger.info(f"Loaded {len(signals)} samples")

    plot_loss_cv_vs_trend(signals, args.output_dir)

    if os.path.exists(args.q3_results):
        with open(args.q3_results) as f:
            q3 = json.load(f)
        plot_auroc_by_epoch(q3, args.output_dir)

    if os.path.exists(args.q4_results):
        with open(args.q4_results) as f:
            q4 = json.load(f)
        plot_noise_type_auroc(q4, args.output_dir)

    plot_loss_trajectories(signals, args.output_dir)

    logger.info(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
