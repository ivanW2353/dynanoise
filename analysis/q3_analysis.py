#!/usr/bin/env python3
"""
Phase 3.5: Q3 Analysis - Signal effectiveness by cumulative epoch windows.
Evaluates how many epochs are needed before P1 signals become useful.
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


def run_q3_analysis(signals: list, output_dir: str) -> dict:
    """Run Q3: Cumulative epoch analysis."""
    results = {}

    for window in ["epochs_1-2", "epochs_1-3", "epochs_1-4", "epochs_1-5"]:
        clean = []
        unlearnable = []

        for s in signals:
            if s["noise_type"] != "clean" and s["noise_type"] != "unlearnable":
                continue

            cv_key = f"{window}_loss_cv"
            trend_key = f"{window}_loss_trend"

            if cv_key in s and trend_key in s:
                if s["noise_type"] == "clean":
                    clean.append(s)
                else:
                    unlearnable.append(s)

        if not clean or not unlearnable:
            logger.warning(f"  {window}: insufficient data")
            continue

        target = clean + unlearnable
        labels = np.array([0 if s["noise_type"] == "clean" else 1 for s in target])

        cv_scores = np.array([s[f"{window}_loss_cv"] for s in target])
        trend_scores = np.array([s[f"{window}_loss_trend"] for s in target])

        # Invert signals (unlearnable has lower cv, more negative trend)
        cv_inv = -cv_scores
        trend_inv = -trend_scores

        cv_norm = (cv_inv - cv_inv.min()) / (cv_inv.max() - cv_inv.min() + 1e-8)
        trend_norm = (trend_inv - trend_inv.min()) / (trend_inv.max() - trend_inv.min() + 1e-8)
        joint = cv_norm + trend_norm

        auroc_cv = roc_auc_score(labels, cv_inv) if len(np.unique(labels)) >= 2 else 0.5
        auroc_joint = roc_auc_score(labels, joint) if len(np.unique(labels)) >= 2 else 0.5

        n_epochs = int(window.split("-")[1])

        logger.info(f"{window} (n_epochs={n_epochs}):")
        logger.info(f"  AUROC (cv):    {auroc_cv:.4f}")
        logger.info(f"  AUROC (joint): {auroc_joint:.4f}")
        logger.info(f"  n_clean: {len(clean)}, n_unlearnable: {len(unlearnable)}")

        results[window] = {
            "n_epochs": n_epochs,
            "auroc_cv": auroc_cv,
            "auroc_joint": auroc_joint,
            "n_clean": len(clean),
            "n_unlearnable": len(unlearnable),
        }

    # Find first epoch where AUROC >= 0.75
    first_usable = None
    for window in ["epochs_1-2", "epochs_1-3", "epochs_1-4", "epochs_1-5"]:
        if window in results:
            if results[window]["auroc_cv"] >= 0.75 or results[window]["auroc_joint"] >= 0.75:
                first_usable = results[window]["n_epochs"]
                break

    logger.info(f"\nFirst epoch with AUROC >= 0.75: epoch {first_usable}")
    logger.info(f"Q3 Pass: first usable at epoch <= 3: {'PASS' if first_usable and first_usable <= 3 else 'FAIL'}")

    results["first_usable_epoch"] = first_usable
    results["pass_min_epoch_3"] = first_usable is not None and first_usable <= 3

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "q3_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Q3 analysis")
    parser.add_argument("--signals-path", default="results/signals.json")
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()

    signals = load_signals(args.signals_path)
    results = run_q3_analysis(signals, args.output_dir)
    logger.info("\nQ3 analysis complete.")


if __name__ == "__main__":
    main()
