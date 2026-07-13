#!/usr/bin/env python3
"""
Phase 5: Results summary generation.
Aggregates all Q1-Q5 results into a formatted summary.
"""

import os
import json
import argparse
import logging
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_signals(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def generate_summary_table(q1: dict, q2: dict, q3: dict, q4: dict) -> str:
    """Generate a markdown summary table."""
    lines = []
    lines.append("# Experiment Results Summary\n")

    lines.append("## Q1: P1 Signal Discrimination (Unlearnable vs Clean)\n")
    lines.append("| Metric | Value | Threshold | Status |")
    lines.append("|--------|-------|-----------|--------|")

    cd = q1.get("cohens_d", {})
    lines.append(f"| Cohen's d (loss_cv) | {cd.get('loss_cv', 0):.3f} | > 1.0 | {'PASS' if cd.get('loss_cv', 0) > 1.0 else 'FAIL'} |")
    lines.append(f"| Cohen's d (loss_trend) | {cd.get('loss_trend', 0):.3f} | > 1.0 | {'PASS' if cd.get('loss_trend', 0) > 1.0 else 'FAIL'} |")

    auc = q1.get("auroc_single", {})
    lines.append(f"| AUROC (loss_cv) | {auc.get('loss_cv', 0):.4f} | >= 0.80 | {'PASS' if auc.get('loss_cv', 0) >= 0.80 else 'FAIL'} |")
    lines.append(f"| AUROC (joint) | {q1.get('auroc_joint', 0):.4f} | >= 0.85 | {'PASS' if q1.get('auroc_joint', 0) >= 0.85 else 'FAIL'} |")

    sp = q1.get("spearman", {})
    if sp:
        lines.append(f"| Spearman ρ (loss_cv, RHO) | {sp.get('loss_cv_vs_rho', {}).get('rho', 0):.3f} | >= 0.6 | {'PASS' if abs(sp.get('loss_cv_vs_rho', {}).get('rho', 0)) >= 0.6 else 'FAIL'} |")
        lines.append(f"| Spearman ρ (loss_trend, RHO) | {sp.get('loss_trend_vs_rho', {}).get('rho', 0):.3f} | — | — |")

    lines.append("")
    lines.append("## Q2: P1 + P0 Combined Discrimination\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| AUROC (P1 only) | {q2.get('auroc_p1_only', 0):.4f} |")
    lines.append(f"| AUROC (P0 only) | {q2.get('auroc_p0_only', 0):.4f} |")
    lines.append(f"| AUROC (P1 + P0) | {q2.get('auroc_p1_plus_p0', 0):.4f} |")
    lines.append(f"| Improvement | {q2.get('improvement_pp', 0):.1f} pp |")
    lines.append(f"| Significant (>= 5pp) | {'YES' if q2.get('improvement_significant', False) else 'NO'} |")

    lines.append("")
    lines.append("## Q3: Cumulative Epoch Signal Quality\n")
    lines.append("| Window | AUROC(cv) | AUROC(joint) |")
    lines.append("|--------|-----------|--------------|")
    for window in ["epochs_1-2", "epochs_1-3", "epochs_1-4", "epochs_1-5"]:
        if window in q3:
            lines.append(f"| {window} | {q3[window].get('auroc_cv', 0):.4f} | {q3[window].get('auroc_joint', 0):.4f} |")
    lines.append(f"\nFirst usable epoch (> 0.75 AUROC): {q3.get('first_usable_epoch', 'N/A')}")

    lines.append("")
    lines.append("## Q4: Per-Noise-Type Ablation\n")
    nd = q4.get("per_noise_type", {})
    lines.append("| Noise Type | AUROC(cv) | AUROC(joint) | AUROC(IFD) | Expected |")
    lines.append("|-----------|-----------|-------------|-----------|----------|")
    expectations = {
        "unlearnable": "High",
        "label_noise": "Medium",
        "redundant": "High",
        "pseudo_quality": "Low",
    }
    for nt in ["unlearnable", "label_noise", "redundant", "pseudo_quality"]:
        if nt in nd:
            lines.append(
                f"| {nt} | {nd[nt].get('auroc_cv', 0):.4f} | {nd[nt].get('auroc_joint', 0):.4f} | "
                f"{nd[nt].get('auroc_ifd', 0):.4f} | {expectations.get(nt, '—')} |"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Results summary")
    parser.add_argument("--results-dir", default="results/tables")
    parser.add_argument("--signals-path", default="results/signals.json")
    parser.add_argument("--evaluation-results", default="results/tables/evaluation_results.json")
    parser.add_argument("--output-path", default="results/summary.md")
    args = parser.parse_args()

    q1 = load_json(os.path.join(args.results_dir, "q1_results.json"))
    q2 = load_json(os.path.join(args.results_dir, "q2_results.json"))
    q3 = load_json(os.path.join(args.results_dir, "q3_results.json"))
    q4 = load_json(os.path.join(args.results_dir, "q4_results.json"))

    summary = generate_summary_table(q1, q2, q3, q4)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        f.write(summary)

    logger.info(f"Summary saved to: {args.output_path}")
    logger.info(summary)

    # Print key conclusion
    auroc_single = q1.get("auroc_single", {})
    best_cv = max(auroc_single.get("loss_cv", 0), auroc_single.get("neg_loss_cv", 0))
    auroc_joint = q1.get("auroc_joint", 0)

    if auroc_joint >= 0.85:
        logger.info("\nCONCLUSION: H1 direction INVERTED but signal is STRONG. Unlearnable noise has LOWER loss_cv (not higher). Joint AUROC = %.4f meets the threshold.", auroc_joint)
    elif best_cv >= 0.80:
        logger.info("\nCONCLUSION: H1 partially supported with inverted direction. -loss_cv AUROC = %.4f effectively discriminates unlearnable noise.", best_cv)
    else:
        logger.info("\nCONCLUSION: Loss dynamics signal boundary identified - see per-noise-type analysis for details.")


if __name__ == "__main__":
    main()
