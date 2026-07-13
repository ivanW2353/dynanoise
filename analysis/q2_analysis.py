#!/usr/bin/env python3
"""
Phase 3.4: Q2 Analysis - P1 + P0 (token_loss_top20) combined discrimination.
Compares AUROC of P1-only vs P1+P0 for unlearnable noise detection.
"""

import os
import json
import argparse
import logging

import numpy as np
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_signals(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def normalize(arr: np.ndarray) -> np.ndarray:
    denom = arr.max() - arr.min()
    if denom < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr.min()) / denom


def run_q2_analysis(signals: list, output_dir: str) -> dict:
    """Run Q2: P1 + P0 combined analysis."""
    clean = [s for s in signals if s["noise_type"] == "clean"]
    unlearnable = [s for s in signals if s["noise_type"] == "unlearnable"]
    target = clean + unlearnable
    labels = np.array([0 if s["noise_type"] == "clean" else 1 for s in target])

    results = {}

    # P1 signals (using correct directions from Q1 findings)
    loss_cv = np.array([s.get("loss_cv", 0) for s in target])
    loss_trend = np.array([s.get("loss_trend", 0) for s in target])

    # Unlearnable has LOWER cv and MORE NEGATIVE trend → use inverted
    p1_score = normalize(-loss_cv) + normalize(-loss_trend)
    auroc_p1 = roc_auc_score(labels, p1_score)

    logger.info(f"AUROC (P1 only, inverted): {auroc_p1:.4f}")

    # P0 signal: token_loss_top20
    token_top20 = np.array([s.get("token_loss_top20", 0) for s in target])

    # For noise A, token_loss_top20 should be LOW (uniform loss distribution → low top20%)
    # For clean, token_loss_top20 should be HIGHER (loss concentrated on hard tokens)
    # So we use (1 - token_top20_norm) as the noise indicator
    t20_norm = normalize(-token_top20)  # Higher values = more noise-like

    auroc_t20 = roc_auc_score(labels, t20_norm)
    logger.info(f"AUROC (token_loss_top20 only): {auroc_t20:.4f}")

    # P1 + P0 combined
    p1p0_score = p1_score + t20_norm
    auroc_p1p0 = roc_auc_score(labels, p1p0_score)

    logger.info(f"AUROC (P1 + P0): {auroc_p1p0:.4f}")
    logger.info(f"Improvement: {(auroc_p1p0 - auroc_p1) * 100:.1f} pp")

    results = {
        "auroc_p1_only": auroc_p1,
        "auroc_p0_only": auroc_t20,
        "auroc_p1_plus_p0": auroc_p1p0,
        "improvement_pp": (auroc_p1p0 - auroc_p1) * 100,
        "improvement_significant": (auroc_p1p0 - auroc_p1) >= 0.05,
    }

    # Feature contribution analysis
    features = {
        "neg_loss_cv": normalize(-loss_cv),
        "neg_loss_trend": normalize(-loss_trend),
        "neg_token_loss_top20": t20_norm,
    }

    for name, scores in features.items():
        try:
            auroc = roc_auc_score(labels, scores)
            logger.info(f"  {name}: AUROC = {auroc:.4f}")
            results[f"auroc_{name}"] = auroc
        except Exception:
            pass

    logger.info(f"\nQ2 Pass/Fail:")
    logger.info(f"  P1+P0 improves >= 5pp over P1: {'PASS' if results['improvement_significant'] else 'FAIL'}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "q2_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Q2 analysis")
    parser.add_argument("--signals-path", default="results/signals.json")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()

    signals = load_signals(args.signals_path)
    results = run_q2_analysis(signals, args.output_dir)
    logger.info("\nQ2 analysis complete.")


if __name__ == "__main__":
    main()
