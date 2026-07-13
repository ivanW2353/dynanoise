#!/usr/bin/env python3
"""
Phase 3.1-3.2: Compute all signals from collected training data.
- P1 signals: loss_mu, loss_sigma, loss_cv, loss_trend (from per-epoch losses)
- P0 signals: token_loss_top20 (from per-token losses at epoch 3)
Also integrates IFD and RHO scores.
"""

import os
import json
import argparse
import logging

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_losses(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_ifd_scores(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_rho_scores(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_token_losses(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
        return data.get("token_losses", {})


def load_dataset_metadata(path: str) -> list:
    """Load metadata from train.jsonl."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def compute_p1_signals(losses_per_sample: dict) -> dict:
    """Compute loss_mu, loss_sigma, loss_cv, loss_trend for each sample."""
    signals = {}

    for idx_str, epoch_losses in losses_per_sample.items():
        loss_values = []
        for epoch_key in sorted(epoch_losses.keys()):
            loss_values.append(epoch_losses[epoch_key])

        loss_values = np.array(loss_values)
        n = len(loss_values)

        if n < 2:
            signals[idx_str] = {
                "loss_mu": float(loss_values[0]) if n == 1 else 0.0,
                "loss_sigma": 0.0,
                "loss_cv": 0.0,
                "loss_trend": 0.0,
            }
            continue

        mu = np.mean(loss_values)
        sigma = np.std(loss_values, ddof=1)
        cv = sigma / mu if mu > 0 else 0.0

        epochs = np.arange(1, n + 1)
        trend = stats.linregress(epochs, loss_values).slope

        signals[idx_str] = {
            "loss_mu": float(mu),
            "loss_sigma": float(sigma),
            "loss_cv": float(cv),
            "loss_trend": float(trend),
        }

    return signals


def compute_cumulative_p1_signals(losses_per_sample: dict, min_epochs: int = 2) -> dict:
    """Compute P1 signals for cumulative epoch windows: 1-2, 1-3, 1-4, 1-5."""
    cumulative = {}

    for idx_str, epoch_losses in losses_per_sample.items():
        loss_values = []
        for epoch_key in sorted(epoch_losses.keys()):
            loss_values.append(epoch_losses[epoch_key])

        cumulative[idx_str] = {}
        for k in range(min_epochs, len(loss_values) + 1):
            vals = np.array(loss_values[:k])
            mu = np.mean(vals)
            sigma = np.std(vals, ddof=1) if k > 1 else 0.0
            cv = sigma / mu if mu > 0 else 0.0
            trend = stats.linregress(np.arange(1, k + 1), vals).slope if k >= 2 else 0.0

            cumulative[idx_str][f"epochs_1-{k}"] = {
                "loss_mu": float(mu),
                "loss_sigma": float(sigma),
                "loss_cv": float(cv),
                "loss_trend": float(trend),
                "n_epochs": k,
            }

    return cumulative


def compute_token_loss_top20(token_losses_per_sample: dict) -> dict:
    """Compute token_loss_top20 for each sample."""
    top20_signals = {}

    for idx_str, token_losses in token_losses_per_sample.items():
        if not token_losses:
            top20_signals[idx_str] = {"token_loss_top20": 0.0}
            continue

        losses = np.array(token_losses)
        total_loss = np.sum(losses)
        if total_loss == 0:
            top20_signals[idx_str] = {"token_loss_top20": 0.0}
            continue

        n_top = max(1, int(np.ceil(len(losses) * 0.2)))
        top_indices = np.argsort(losses)[-n_top:]
        top_loss = np.sum(losses[top_indices])

        top20_signals[idx_str] = {
            "token_loss_top20": float(top_loss / total_loss),
            "n_tokens": int(len(losses)),
            "mean_token_loss": float(np.mean(losses)),
            "std_token_loss": float(np.std(losses)),
        }

    return top20_signals


def normalize_signal(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    vmin, vmax = np.min(values), np.max(values)
    if vmax - vmin < 1e-8:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def compute_composite_score(p1_signals: dict, alpha: float = 0.5) -> dict:
    """Composite score = alpha * (-cv_norm) + (1-alpha) * (-trend_norm).
    Higher score = more noise-like. Uses inverted CV since unlearnable has LOWER cv than clean."""
    cv_values = np.array([v["loss_cv"] for v in p1_signals.values()])
    trend_values = np.array([v["loss_trend"] for v in p1_signals.values()])

    cv_norm = normalize_signal(-cv_values)
    trend_norm = normalize_signal(-trend_values)

    scores = alpha * cv_norm + (1 - alpha) * trend_norm

    result = {}
    for i, idx in enumerate(p1_signals.keys()):
        result[idx] = {
            "composite_score": float(scores[i]),
            "cv_norm": float(cv_norm[i]),
            "trend_norm": float(trend_norm[i]),
        }

    return result


def compute_zscore_composite_score(
    p1_signals: dict,
    token_top20: dict,
    ifd_scores: dict,
    dataset_metadata: list,
) -> dict:
    """Phase 5 composite: absolute z-score deviation from clean mean.
    Captures deviation in EITHER direction — works for both Noise A (low CV/top20, high IFD)
    and Noise E (high CV/top20, low IFD).

    noise_score = |z(token_loss_top20)| + |z(ifd)| + |z(loss_cv)|
    """
    clean_idx = set(str(i) for i, s in enumerate(dataset_metadata) if s.get("noise_type") == "clean")

    def get_vals(signal_dict, key, idx_list):
        return np.array([float(signal_dict[i][key]) for i in idx_list if i in signal_dict])

    clean_keys = sorted(clean_idx)

    clean_cv = get_vals(p1_signals, "loss_cv", clean_keys)
    clean_t20 = get_vals(token_top20, "token_loss_top20", clean_keys)
    clean_ifd = get_vals(ifd_scores, "ifd", clean_keys)

    # Clean distribution stats
    cv_mean, cv_std = np.mean(clean_cv), np.std(clean_cv) or 1e-8
    t20_mean, t20_std = np.mean(clean_t20), np.std(clean_t20) or 1e-8
    ifd_mean, ifd_std = np.mean(clean_ifd), np.std(clean_ifd) or 1e-8

    logger.info(f"Z-score composite: clean cv={cv_mean:.4f}±{cv_std:.4f}, "
                f"t20={t20_mean:.4f}±{t20_std:.4f}, ifd={ifd_mean:.4f}±{ifd_std:.4f}")

    result = {}
    all_idx = set(p1_signals.keys())
    for idx in sorted(all_idx, key=int):
        z_cv = ((p1_signals[idx].get("loss_cv", 0) - cv_mean) / cv_std)
        z_t20 = ((token_top20.get(idx, {}).get("token_loss_top20", 0) - t20_mean) / t20_std)
        z_ifd = ((ifd_scores.get(idx, {}).get("ifd", 0) - ifd_mean) / ifd_std)

        score = abs(z_cv) + abs(z_t20) + abs(z_ifd)

        result[idx] = {
            "composite_score": float(score),
            "z_cv": float(z_cv),
            "z_token_top20": float(z_t20),
            "z_ifd": float(z_ifd),
        }

    return result


def merge_all_signals(
    dataset: list,
    p1_signals: dict,
    token_top20: dict,
    ifd_scores: dict,
    rho_scores: dict,
    cumulative_signals: dict,
    composite_scores: dict,
) -> list:
    """Merge all signals into per-sample records."""
    merged = []

    for i, sample in enumerate(dataset):
        idx_str = str(i)
        record = {
            "idx": i,
            "source_idx": sample.get("source_idx", i),
            "noise_type": sample.get("noise_type", "clean"),
            "is_noise": sample.get("is_noise", False),
            "generated_by": sample.get("generated_by", "human"),
            "category": sample.get("category", ""),
            "instruction_preview": sample.get("instruction", "")[:100],
        }

        if idx_str in p1_signals:
            record.update(p1_signals[idx_str])

        if idx_str in token_top20:
            record.update(token_top20[idx_str])

        if idx_str in ifd_scores:
            record["ifd"] = ifd_scores[idx_str].get("ifd", 0.0)
            record["conditional_loss"] = ifd_scores[idx_str].get("conditional_loss", 0.0)
            record["unconditional_loss"] = ifd_scores[idx_str].get("unconditional_loss", 0.0)

        if idx_str in rho_scores:
            record["rho_score"] = rho_scores[idx_str].get("rho_score", 0.0)
            record["loss_main"] = rho_scores[idx_str].get("loss_main", 0.0)
            record["loss_holdout"] = rho_scores[idx_str].get("loss_holdout", 0.0)

        if idx_str in composite_scores:
            record.update(composite_scores[idx_str])

        if idx_str in cumulative_signals:
            for window, signals in cumulative_signals[idx_str].items():
                for k, v in signals.items():
                    record[f"{window}_{k}"] = v

        merged.append(record)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Compute all signals")
    parser.add_argument(
        "--losses-path",
        default="checkpoints/main_model/all_losses.json",
        help="Path to all_losses.json from main model training",
    )
    parser.add_argument(
        "--token-losses-path",
        default="checkpoints/main_model/token_losses_epoch_3.json",
        help="Path to token_losses_epoch_3.json",
    )
    parser.add_argument(
        "--ifd-path",
        default="checkpoints/ifd_scores.json",
        help="Path to IFD scores",
    )
    parser.add_argument(
        "--rho-path",
        default="checkpoints/rho_scores.json",
        help="Path to RHO scores",
    )
    parser.add_argument(
        "--data-path",
        default="data/train.jsonl",
        help="Path to training data with metadata",
    )
    parser.add_argument(
        "--output-path",
        default="results/signals.csv",
        help="Output path for merged signals CSV",
    )
    parser.add_argument(
        "--output-json",
        default="results/signals.json",
        help="Output path for merged signals JSON",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for -loss_cv in composite score (1.0 = pure cv, 0.0 = pure trend).",
    )
    parser.add_argument(
        "--composite-mode",
        default="cv_trend",
        choices=["cv_trend", "zscore"],
        help="Composite score mode: cv_trend (default) or zscore (Phase 5, detects noise in both directions)",
    )
    args = parser.parse_args()

    logger.info("Loading data...")
    dataset = load_dataset_metadata(args.data_path)
    all_losses = load_all_losses(args.losses_path)

    p1_signals = compute_p1_signals(all_losses)
    logger.info(f"Computed P1 signals for {len(p1_signals)} samples")

    cumulative_signals = compute_cumulative_p1_signals(all_losses)
    logger.info("Computed cumulative P1 signals")

    token_losses = load_token_losses(args.token_losses_path) if os.path.exists(args.token_losses_path) else {}
    token_top20 = compute_token_loss_top20(token_losses)
    logger.info(f"Computed token_loss_top20 for {len(token_top20)} samples")

    ifd_scores = load_ifd_scores(args.ifd_path) if os.path.exists(args.ifd_path) else {}
    rho_scores = load_rho_scores(args.rho_path) if os.path.exists(args.rho_path) else {}

    if args.composite_mode == "zscore":
        logger.info("Using zscore composite (absolute deviation from clean mean)")
        composite_scores = compute_zscore_composite_score(
            p1_signals, token_top20, ifd_scores, dataset
        )
    else:
        composite_scores = compute_composite_score(p1_signals, alpha=args.alpha)

    merged = merge_all_signals(
        dataset, p1_signals, token_top20, ifd_scores, rho_scores, cumulative_signals, composite_scores
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    import csv
    all_keys = sorted(set().union(*(d.keys() for d in merged)))
    with open(args.output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(merged)

    with open(args.output_json, "w") as f:
        json.dump(merged, f, indent=2)

    logger.info(f"Saved {len(merged)} records to {args.output_path} and {args.output_json}")

    # Summary statistics
    for nt in ["clean", "unlearnable", "label_noise", "redundant", "pseudo_quality", "shortcut"]:
        subset = [s for s in merged if s["noise_type"] == nt]
        if not subset:
            continue
        cvs = [s.get("loss_cv", 0) for s in subset]
        trends = [s.get("loss_trend", 0) for s in subset]
        logger.info(f"\n{nt} ({len(subset)} samples):")
        logger.info(f"  loss_cv:   mean={np.mean(cvs):.4f}, std={np.std(cvs):.4f}")
        logger.info(f"  loss_trend: mean={np.mean(trends):.4f}, std={np.std(trends):.4f}")


if __name__ == "__main__":
    main()
