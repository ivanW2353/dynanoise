#!/usr/bin/env python3
"""
Phase 6B: Extract top/bottom samples by token_loss_top20 for manual inspection.
"""

import json, os, csv, argparse, logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 6B: Extract samples for manual inspection")
    parser.add_argument("--signals-path", required=True, help="Path to signals JSON")
    parser.add_argument("--data-path", required=True, help="Path to training data JSONL")
    parser.add_argument("--output-path", default="results/tables_p6/inspection_samples.csv")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--signal", default="token_loss_top20", choices=["token_loss_top20", "ifd", "composite_score"])
    parser.add_argument("--direction", default="asc", choices=["asc", "desc"],
                        help="asc = low signal first (most noise-like for token_top20)")
    args = parser.parse_args()

    with open(args.signals_path) as f:
        signals = json.load(f)

    with open(args.data_path) as f:
        dataset = [json.loads(line) for line in f]

    signal_map = {}
    for s in signals:
        signal_map[s["idx"]] = s.get(args.signal, 0)

    # Sort
    paired = [(i, signal_map.get(i, 0), dataset[i]) for i in range(min(len(dataset), len(signals)))]
    paired.sort(key=lambda x: x[1], reverse=(args.direction == "desc"))

    # Top + Bottom
    top_n = paired[:args.n_samples]
    bottom_n = paired[-args.n_samples:] if len(paired) >= args.n_samples * 2 else []

    all_samples = []
    for label, samples in [("Top (noise-like)", top_n), ("Bottom (high-quality)", bottom_n)]:
        for idx, score, sample in samples:
            all_samples.append({
                "group": label,
                "index": idx,
                "signal_score": round(score, 4),
                "instruction": sample.get("instruction", "")[:300],
                "response": sample.get("response", "")[:300],
                "noise_type": sample.get("noise_type", "unknown"),
                "human_label": "",
            })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    fieldnames = ["group", "index", "signal_score", "noise_type", "instruction", "response", "human_label"]
    with open(args.output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_samples)

    logger.info(f"Saved {len(all_samples)} samples to {args.output_path}")
    logger.info(f"  Top {args.n_samples} (low {args.signal}): mean score={np.mean([s[1] for s in top_n]):.4f}")
    if bottom_n:
        logger.info(f"  Bottom {args.n_samples}: mean score={np.mean([s[1] for s in bottom_n]):.4f}")
    logger.info(f"\nOpen the CSV and annotate 'human_label' column:")
    logger.info(f"  - truncated (截断)")
    logger.info(f"  - format_error (格式错)")
    logger.info(f"  - factual_error (事实错)")
    logger.info(f"  - off_topic (流畅但离题)")
    logger.info(f"  - hallucination (幻觉)")
    logger.info(f"  - normal (正常/高质量)")


if __name__ == "__main__":
    main()
