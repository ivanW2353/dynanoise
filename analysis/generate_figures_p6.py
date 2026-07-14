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

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.correlation_path) as f:
        corr = json.load(f)

    # ── Bar chart: correlation with avg_rating ──
    rho_t20 = corr.get("avg_rating", {}).get("rho_token_top20", 0)
    rho_ifd = corr.get("avg_rating", {}).get("rho_ifd", 0)
    p_t20 = corr.get("avg_rating", {}).get("p_token_top20", 1)
    p_ifd = corr.get("avg_rating", {}).get("p_ifd", 1)
    n_valid = corr.get("avg_rating", {}).get("n", 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    bars = ax.bar(x, [rho_t20, rho_ifd], color=["#2E86AB", "#F18F01"], width=0.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    for bar, rho, p in zip(bars, [rho_t20, rho_ifd], [p_t20, p_ifd]):
        y_pos = rho + 0.03 if rho >= 0 else rho - 0.08
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"ρ={rho:.3f}\np={p:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["token_loss_top20", "IFD"], fontsize=11)
    ax.set_ylabel("Spearman ρ with DIBT avg_rating", fontsize=12)
    ax.set_title(f"Phase 6A: Signal-Human Quality Correlation\n(DIBT, n={n_valid})", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "dibt_correlation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ρ(token_top20)={rho_t20:.3f}, ρ(IFD)={rho_ifd:.3f}, n={n_valid}")
    logger.info(f"Figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
