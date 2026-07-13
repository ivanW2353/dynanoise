#!/usr/bin/env python3
"""
Phase 4.6: Manual inspection helper.
Samples dropped items from P1-filtered and RHO-filtered groups for human annotation.
Outputs a formatted file for manual labeling.
"""

import os
import json
import random
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_signals(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def load_dataset(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def sample_dropped(
    dataset: list,
    signals: list,
    strategy: str,
    n_samples: int = 50,
    drop_ratio: float = 0.10,
) -> list:
    """Sample items that would be dropped by given strategy."""
    signal_map = {}
    for s in signals:
        signal_map[s["idx"]] = s

    if strategy == "p1_filtered":
        scores = [(i, signal_map[i].get("composite_score", 0)) for i in range(len(dataset))]
        scores.sort(key=lambda x: x[1], reverse=True)
    elif strategy == "rho_filtered":
        scores = [(i, signal_map[i].get("rho_score", 0)) for i in range(len(dataset))]
        scores.sort(key=lambda x: x[1])
    elif strategy == "ifd_only":
        scores = [(i, signal_map[i].get("ifd", 0)) for i in range(len(dataset))]
        scores.sort(key=lambda x: x[1])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    n_drop = int(len(dataset) * drop_ratio)
    dropped_indices = [idx for idx, _ in scores[:n_drop]]

    sampled = random.sample(dropped_indices, min(n_samples, len(dropped_indices)))

    results = []
    for idx in sampled:
        s = dataset[idx]
        result = {
            "index": idx,
            "instruction": s.get("instruction", ""),
            "response": s.get("response", ""),
            "noise_type": s.get("noise_type", "unknown"),
            "is_noise": s.get("is_noise", False),
            "strategy": strategy,
            "score": signal_map[idx].get("composite_score", signal_map[idx].get("rho_score", 0)),
            "human_label": "",
        }
        results.append(result)

    return results


def generate_inspection_csv(samples: dict, output_path: str):
    """Generate a CSV file for human annotation."""
    import csv

    all_samples = []
    for strategy, items in samples.items():
        all_samples.extend(items)

    fieldnames = [
        "index", "strategy", "noise_type", "is_noise", "score",
        "instruction", "response", "human_label", "notes",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in all_samples:
            row = {k: s.get(k, "") for k in fieldnames}
            writer.writerow(row)

    logger.info(f"Saved {len(all_samples)} samples for manual inspection to: {output_path}")


def compute_inspection_metrics(csv_path: str):
    """Compute precision/recall from human-labeled inspection results."""
    import csv

    labels = {"p1_filtered": [], "rho_filtered": []}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            strategy = row.get("strategy", "")
            human_label = row.get("human_label", "").strip().lower()
            if strategy in labels:
                labels[strategy].append(human_label)

    for strategy, lbls in labels.items():
        if not lbls:
            logger.warning(f"No labels for {strategy}")
            continue

        n_confirmed = sum(1 for l in lbls if l in ("确实该丢弃", "confirmed", "yes", "true"))
        n_false_positive = sum(1 for l in lbls if l in ("误伤", "false_positive", "no", "false"))
        n_ambiguous = sum(1 for l in lbls if l in ("难判断", "ambiguous", "maybe"))
        n_unlabeled = sum(1 for l in lbls if not l or l == "")

        total_labeled = n_confirmed + n_false_positive + n_ambiguous
        precision = n_confirmed / max(total_labeled, 1)

        logger.info(f"\n{strategy}:")
        logger.info(f"  Total samples: {len(lbls)}")
        logger.info(f"  Confirmed noise: {n_confirmed}")
        logger.info(f"  False positive (误伤): {n_false_positive}")
        logger.info(f"  Ambiguous: {n_ambiguous}")
        logger.info(f"  Unlabeled: {n_unlabeled}")
        logger.info(f"  Precision: {precision:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Manual inspection helper")
    parser.add_argument(
        "--data-path",
        default="data/train.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--signals-path",
        default="results/signals.json",
        help="Path to computed signals",
    )
    parser.add_argument(
        "--output-path",
        default="results/tables/inspection_samples.csv",
        help="Output CSV for manual annotation",
    )
    parser.add_argument(
        "--n-samples", type=int, default=50, help="Samples per strategy"
    )
    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        help="Compute metrics from labeled CSV (requires --csv-path)",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Path to human-labeled CSV for metric computation",
    )
    args = parser.parse_args()

    if args.compute_metrics:
        if not args.csv_path:
            logger.error("--csv-path required for --compute-metrics")
            return
        compute_inspection_metrics(args.csv_path)
        return

    random.seed(42)

    dataset = load_dataset(args.data_path)
    signals = load_signals(args.signals_path)

    p1_samples = sample_dropped(dataset, signals, "p1_filtered", args.n_samples)
    rho_samples = sample_dropped(dataset, signals, "rho_filtered", args.n_samples)

    generate_inspection_csv(
        {"p1_filtered": p1_samples, "rho_filtered": rho_samples},
        args.output_path,
    )

    logger.info(f"\nTotal dropped by P1 (top 10%): {int(len(dataset) * 0.10)}")
    logger.info(f"Sampled for inspection: {len(p1_samples)} (P1) + {len(rho_samples)} (RHO)")
    logger.info(f"\nPlease annotate the 'human_label' column as:")
    logger.info(f"  - 'confirmed' (确实该丢弃)")
    logger.info(f"  - 'false_positive' (误伤)")
    logger.info(f"  - 'ambiguous' (难判断)")
    logger.info(f"\nThen run with --compute-metrics --csv-path {args.output_path}")


if __name__ == "__main__":
    main()
