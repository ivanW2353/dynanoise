#!/usr/bin/env python3
"""
Phase 3.3: Q1 Analysis - P1 signal discrimination between unlearnable noise and clean data.
Evaluates: Cohen's d, AUROC, Spearman correlation with RHO.
"""

import os
import json
import argparse
import logging

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_signals(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-8:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute AUROC for binary classification."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_scores)


def run_q1_analysis(signals: list, output_dir: str) -> dict:
    """Run Q1: P1 signals vs unlearnable noise."""
    clean = [s for s in signals if s["noise_type"] == "clean"]
    unlearnable = [s for s in signals if s["noise_type"] == "unlearnable"]

    target = clean + unlearnable
    labels = np.array([0 if s["noise_type"] == "clean" else 1 for s in target])

    results = {}

    # Cohen's d
    clean_cv = np.array([s.get("loss_cv", 0) for s in clean])
    unlearnable_cv = np.array([s.get("loss_cv", 0) for s in unlearnable])
    clean_trend = np.array([s.get("loss_trend", 0) for s in clean])
    unlearnable_trend = np.array([s.get("loss_trend", 0) for s in unlearnable])

    d_cv = cohens_d(unlearnable_cv, clean_cv)
    d_trend = cohens_d(unlearnable_trend, clean_trend)

    logger.info(f"Cohen's d (loss_cv):   {d_cv:.4f}")
    logger.info(f"Cohen's d (loss_trend): {d_trend:.4f}")

    results["cohens_d"] = {"loss_cv": d_cv, "loss_trend": d_trend}

    # AUROC - single signals (test both directions)
    cv_scores = np.array([s.get("loss_cv", 0) for s in target])
    trend_scores = np.array([s.get("loss_trend", 0) for s in target])

    auroc_cv = compute_auroc(labels, cv_scores)
    auroc_cv_inv = compute_auroc(labels, -cv_scores)
    auroc_trend = compute_auroc(labels, trend_scores)
    auroc_trend_inv = compute_auroc(labels, -trend_scores)

    # Pick the better direction for each
    best_cv = max(auroc_cv, auroc_cv_inv)
    best_trend = max(auroc_trend, auroc_trend_inv)

    logger.info(f"\nAUROC (loss_cv):            {auroc_cv:.4f}")
    logger.info(f"AUROC (-loss_cv):           {auroc_cv_inv:.4f}")
    logger.info(f"AUROC (loss_trend):          {auroc_trend:.4f}")
    logger.info(f"AUROC (-loss_trend):         {auroc_trend_inv:.4f}")

    results["auroc_single"] = {
        "loss_cv": float(auroc_cv),
        "neg_loss_cv": float(auroc_cv_inv),
        "loss_trend": float(auroc_trend),
        "neg_loss_trend": float(auroc_trend_inv),
    }

    # AUROC - joint signals in the right direction
    cv_signed = cv_scores if auroc_cv >= auroc_cv_inv else -cv_scores
    trend_signed = -trend_scores if auroc_trend_inv >= auroc_trend else trend_scores

    cv_norm = (cv_signed - cv_signed.min()) / (cv_signed.max() - cv_signed.min() + 1e-8)
    trend_norm = (trend_signed - trend_signed.min()) / (trend_signed.max() - trend_signed.min() + 1e-8)
    joint_score = cv_norm + trend_norm

    auroc_joint = compute_auroc(labels, joint_score)
    logger.info(f"AUROC (joint cv + trend):  {auroc_joint:.4f}")

    results["auroc_joint"] = float(auroc_joint)

    # Spearman correlation with RHO (use best direction)
    rho_values = np.array([s.get("rho_score", 0) for s in target])
    if np.any(rho_values != 0):
        rho_cv, p_cv = spearmanr(cv_signed, rho_values)
        rho_trend, p_trend = spearmanr(trend_signed, rho_values)

        logger.info(f"\nSpearman ρ (best_cv, rho_score):      {rho_cv:.4f} (p={p_cv:.4f})")
        logger.info(f"Spearman ρ (best_trend, rho_score):    {rho_trend:.4f} (p={p_trend:.4f})")

        results["spearman"] = {
            "best_cv_vs_rho": {"rho": float(rho_cv), "p_value": float(p_cv)},
            "best_trend_vs_rho": {"rho": float(rho_trend), "p_value": float(p_trend)},
        }

    # Pass/fail
    threshold_cv = best_cv >= 0.80
    threshold_joint = auroc_joint >= 0.85
    threshold_d = abs(d_cv) > 1.0

    results["cohens_d"] = {"loss_cv": float(d_cv), "loss_trend": float(d_trend)}
    results["pass_fail"] = {
        "auroc_cv_ge_0.80": bool(threshold_cv),
        "auroc_joint_ge_0.85": bool(threshold_joint),
        "cohens_d_gt_1.0": bool(threshold_d),
    }

    logger.info(f"\nQ1 Pass/Fail Summary:")
    logger.info(f"  AUROC(best cv) >= 0.80:  {'PASS' if threshold_cv else 'FAIL'}")
    logger.info(f"  AUROC(joint) >= 0.85:    {'PASS' if threshold_joint else 'FAIL'}")
    logger.info(f"  |Cohen's d(cv)| > 1.0:   {'PASS' if threshold_d else 'FAIL'}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "q1_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Q1 analysis")
    parser.add_argument(
        "--signals-path",
        default="results/signals.json",
        help="Path to signals JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables",
        help="Output directory for results",
    )
    args = parser.parse_args()

    signals = load_signals(args.signals_path)

    clean_count = sum(1 for s in signals if s["noise_type"] == "clean")
    unlearnable_count = sum(1 for s in signals if s["noise_type"] == "unlearnable")
    logger.info(f"Clean samples: {clean_count}")
    logger.info(f"Unlearnable samples: {unlearnable_count}")
    logger.info(f"Total samples: {len(signals)}")

    results = run_q1_analysis(signals, args.output_dir)
    logger.info("\nQ1 analysis complete. Results saved.")


if __name__ == "__main__":
    main()
