#!/usr/bin/env python3
"""
Phase 3.6: Q4 Ablation - P1 signal discrimination for each noise type vs clean.
Creates a per-noise-type AUROC matrix and confusion analysis.
Also includes IFD comparison as baseline.
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


def compute_pairwise_auroc(clean: list, noise: list, signal_name: str) -> float:
    """Compute AUROC for clean vs noise using a single signal, best direction."""
    target = clean + noise
    labels = np.array([0 if s["noise_type"] == "clean" else 1 for s in target])
    scores = np.array([s.get(signal_name, 0) for s in target])

    if len(np.unique(labels)) < 2:
        return 0.5

    auroc_1 = roc_auc_score(labels, scores)
    auroc_2 = roc_auc_score(labels, -scores)
    return max(auroc_1, auroc_2)


def compute_pairwise_auroc_joint(clean: list, noise: list) -> float:
    """Compute AUROC using joint P1 signals (cv + trend), best direction."""
    target = clean + noise
    labels = np.array([0 if s["noise_type"] == "clean" else 1 for s in target])

    cv = np.array([s.get("loss_cv", 0) for s in target])
    trend = np.array([s.get("loss_trend", 0) for s in target])

    # Test both directions and pick best
    joint_1 = normalize(cv) + normalize(-trend)
    joint_2 = normalize(-cv) + normalize(-trend)

    if len(np.unique(labels)) < 2:
        return 0.5

    auroc_1 = roc_auc_score(labels, joint_1)
    auroc_2 = roc_auc_score(labels, joint_2)
    return max(auroc_1, auroc_2)


def run_q4_analysis(signals: list, output_dir: str) -> dict:
    """Run Q4: Per-noise-type ablation."""
    clean = [s for s in signals if s["noise_type"] == "clean"]
    noise_types = ["unlearnable", "label_noise", "redundant", "pseudo_quality"]

    results = {"per_noise_type": {}, "matrix": {}}

    logger.info(f"Clean baseline: {len(clean)} samples")
    logger.info(f"\n{'Noise Type':<20} {'AUROC(cv)':>10} {'AUROC(joint)':>12} {'AUROC(IFD)':>10}")
    logger.info("-" * 55)

    for nt in noise_types:
        noise_samples = [s for s in signals if s["noise_type"] == nt]
        if not noise_samples:
            logger.warning(f"  {nt}: no samples found")
            continue

        auroc_cv = compute_pairwise_auroc(clean, noise_samples, "loss_cv")
        auroc_joint = compute_pairwise_auroc_joint(clean, noise_samples)
        auroc_ifd = compute_pairwise_auroc(clean, noise_samples, "ifd")

        logger.info(
            f"{nt:<20} {auroc_cv:>10.4f} {auroc_joint:>12.4f} {auroc_ifd:>10.4f}"
        )

        results["per_noise_type"][nt] = {
            "auroc_cv": auroc_cv,
            "auroc_joint": auroc_joint,
            "auroc_ifd": auroc_ifd,
            "n_samples": len(noise_samples),
        }

    # Multi-class confusion matrix (one-vs-rest)
    all_clean = [s for s in signals if s["noise_type"] == "clean"][:len(clean)]
    for nt_a in noise_types:
        results["matrix"][nt_a] = {}
        for nt_b in noise_types:
            if nt_a == nt_b:
                results["matrix"][nt_a][nt_b] = 1.0
                continue
            samples_a = [s for s in signals if s["noise_type"] == nt_a]
            samples_b = [s for s in signals if s["noise_type"] == nt_b]
            if not samples_a or not samples_b:
                results["matrix"][nt_a][nt_b] = 0.5
                continue
            auroc = compute_pairwise_auroc_joint(samples_b, samples_a)
            results["matrix"][nt_a][nt_b] = auroc

    # Expected boundaries
    logger.info(f"\nQ4 Expected Results:")
    logger.info(f"  A (unlearnable): AUROC(joint) should be HIGH")
    logger.info(f"  B (label_noise): AUROC(joint) should be MEDIUM")
    logger.info(f"  C (redundant):   AUROC(joint) should be HIGH")
    logger.info(f"  D (pseudo_quality): AUROC(joint) should be LOW (near 0.5)")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "q4_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Q4 ablation analysis")
    parser.add_argument("--signals-path", default="results/signals.json")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()

    signals = load_signals(args.signals_path)
    results = run_q4_analysis(signals, args.output_dir)
    logger.info("\nQ4 analysis complete.")


if __name__ == "__main__":
    main()
