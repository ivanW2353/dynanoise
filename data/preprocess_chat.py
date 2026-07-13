#!/usr/bin/env python3
"""
Preprocess chat datasets into JSONL format for train_main.py.
Converts multi-turn conversations into instruction-response pairs.
Supports: ShareGPT, WildChat, lmsys-chat-1m.
"""

import json, argparse, logging
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASETS = {
    "sharegpt": "anon8231489123/ShareGPT_Vicuna_unfiltered",
    "wildchat": "allenai/WildChat",
    "lmsys": "lmsys/lmsys-chat-1m",
}


def extract_sharegpt_pairs(example: dict) -> list:
    """ShareGPT format: conversations is a list of {'from': ..., 'value': ...}."""
    pairs = []
    conversations = example.get("conversations", [])
    if isinstance(conversations, list) and len(conversations) >= 2:
        for i in range(0, len(conversations) - 1, 2):
            human = conversations[i].get("value", "")
            gpt = conversations[i + 1].get("value", "") if i + 1 < len(conversations) else ""
            if human.strip() and gpt.strip():
                pairs.append({"instruction": human.strip(), "response": gpt.strip()})
    return pairs


def extract_lmsys_pairs(example: dict) -> list:
    """lmsys-chat-1m format: conversation is list of {'role': 'user'/'assistant', 'content': ...}."""
    pairs = []
    conversation = example.get("conversation", [])
    if not isinstance(conversation, list):
        return pairs
    for i in range(len(conversation) - 1):
        turn_a = conversation[i]
        turn_b = conversation[i + 1]
        if turn_a.get("role") == "user" and turn_b.get("role") == "assistant":
            human = turn_a.get("content", "")
            gpt = turn_b.get("content", "")
            if human.strip() and gpt.strip():
                pairs.append({"instruction": human.strip(), "response": gpt.strip()})
    return pairs


def extract_wildchat_pairs(example: dict) -> list:
    """WildChat format: same as lmsys, conversation with role/content."""
    return extract_lmsys_pairs(example)


EXTRACTORS = {
    "sharegpt": extract_sharegpt_pairs,
    "wildchat": extract_wildchat_pairs,
    "lmsys": extract_lmsys_pairs,
}


def main():
    parser = argparse.ArgumentParser(description="Preprocess chat datasets to JSONL")
    parser.add_argument("--dataset", default="lmsys", choices=list(DATASETS.keys()),
                        help="Dataset to preprocess")
    parser.add_argument("--output", default="data/sharegpt_train.jsonl", help="Output JSONL path")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max instruction-response pairs to extract")
    parser.add_argument("--max-examples", type=int, default=None, help="Max raw conversation examples to process")
    args = parser.parse_args()

    dataset_name = DATASETS[args.dataset]
    extract_fn = EXTRACTORS[args.dataset]
    logger.info(f"Loading {dataset_name} (split={args.split})")

    ds = load_dataset(dataset_name, split=args.split, streaming=True)
    logger.info(f"Streaming dataset...")

    all_pairs = []
    for example in tqdm(ds, desc="Extracting pairs"):
        pairs = extract_fn(example)
        for p in pairs:
            all_pairs.append(p)
            if args.max_samples and len(all_pairs) >= args.max_samples:
                break
        if args.max_samples and len(all_pairs) >= args.max_samples:
            break
        if args.max_examples and len(all_pairs) // 2 >= args.max_examples:
            break

    if args.max_samples:
        all_pairs = all_pairs[:args.max_samples]

    with open(args.output, "w") as f:
        for p in all_pairs:
            p["noise_type"] = "clean"
            p["is_noise"] = False
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(all_pairs)} instruction-response pairs to {args.output}")


if __name__ == "__main__":
    main()
