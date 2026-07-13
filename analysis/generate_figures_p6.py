#!/usr/bin/env python3
"""
Phase 6: Generate figures — DIBT correlation radar chart and bar chart.
"""

import json, os, argparse, logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIBT_DIMENSIONS = [
    "fluency", "coherence", "factuality", "relevance", "completeness",
    "conciseness", "helpfulness", "safety", "diversity", "overall",
]


def main():
    parser = argparse.ArgumentParser(description="Phase 6 figure generation")
    parser.add_argument("--correlation-path", default="results/tables_p6/dibt_correlation.json")
    parser.add_argument("--output-dir", default="results/figures_p6")
    args = parser.parse_args()

    with open(args.correlation_path) as f:
        corr = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract values
    dims = [d for d in DIBT_DIMENSIONS if d in corr]
    rho_t20 = [corr[d]["rho_token_top20"] for d in dims]
    rho_ifd = [corr[d]["rho_ifd"] for d in dims]

    # ── 1. Radar chart ──
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]
    rho_t20_closed = rho_t20 + rho_t20[:1]
    rho_ifd_closed = rho_ifd + rho_ifd[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, rho_t20_closed, "o-", color="#2E86AB", linewidth=2, label="token_loss_top20")
    ax.fill(angles, rho_t20_closed, alpha=0.1, color="#2E86AB")
    ax.plot(angles, rho_ifd_closed, "s-", color="#F18F01", linewidth=2, label="IFD")
    ax.fill(angles, rho_ifd_closed, alpha=0.1, color="#F18F01")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_title("Phase 6: Spearman ρ — Loss Dynamics vs DIBT Quality Dimensions", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_ylim(-0.5, 0.6)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    fig.savefig(os.path.join(args.output_dir, "dibt_radar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Bar chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dims))
    w = 0.35
    ax.bar(x - w/2, rho_t20, w, label="token_loss_top20", color="#2E86AB")
    ax.bar(x + w/2, rho_ifd, w, label="IFD", color="#F18F01")
    for i, (a, b) in enumerate(zip(rho_t20, rho_ifd)):
        ax.text(i - w/2, a + 0.02 if a >= 0 else a - 0.06, f"{a:.3f}", ha="center", fontsize=7)
        ax.text(i + w/2, b + 0.02 if b >= 0 else b - 0.06, f"{b:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_title("Phase 6: Signal-Human Quality Correlation (DIBT)", fontsize=13)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "dibt_bars.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
