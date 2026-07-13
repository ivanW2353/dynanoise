#!/usr/bin/env python3
"""
Phase 1: Data preparation for loss dynamics noise detection experiment.
Constructs a mixed dataset with known noise type labels.
"""

import os
import sys
import json
import random
import argparse
import logging
from collections import defaultdict

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from datasets import load_dataset, Dataset, concatenate_datasets
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)
np.random.seed(42)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_dolly(split: str = "train") -> Dataset:
    """Download databricks-dolly-15k dataset."""
    return load_dataset("databricks/databricks-dolly-15k", split=split)


def split_dataset(dataset: Dataset, config: dict) -> dict:
    """Randomly split into train/val/test. Returns datasets and original indices."""
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    n_train = config["data"]["train_size"]
    n_val = config["data"]["val_size"]
    n_test = config["data"]["test_size"]

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val : n_train + n_val + n_test]

    return {
        "train": dataset.select(train_indices),
        "val": dataset.select(val_indices),
        "test": dataset.select(test_indices),
    }, {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def add_base_metadata(sample: dict, idx: int) -> dict:
    """Add base metadata fields."""
    return {
        "instruction": sample["instruction"],
        "response": sample["response"],
        "original_response": sample["response"],
        "context": sample.get("context", ""),
        "category": sample.get("category", ""),
        "noise_type": "clean",
        "is_noise": False,
        "source_idx": idx,
        "generated_by": "human",
    }


def construct_noise_a_random_tokens(
    clean_samples: list, config: dict, tokenizer
) -> list:
    """Noise Type A: Unlearnable noise via random token sequences."""
    num_noise = int(len(clean_samples) * config["data"]["noise_ratios"]["unlearnable"])
    selected = random.sample(clean_samples, num_noise)

    vocab_tokens = list(range(len(tokenizer)))
    noise_samples = []

    for s in tqdm(selected, desc="Constructing Noise A (random tokens)"):
        original_tokens = tokenizer.encode(s["response"])
        n = len(original_tokens)
        random_tokens = random.choices(vocab_tokens, k=n)
        noise_response = tokenizer.decode(random_tokens)

        noise_samples.append(
            {
                "instruction": s["instruction"],
                "response": noise_response,
                "original_response": s["response"],
                "context": s.get("context", ""),
                "category": s.get("category", ""),
                "noise_type": "unlearnable",
                "is_noise": True,
                "source_idx": s["source_idx"],
                "generated_by": "random_sampler",
            }
        )

    return noise_samples


def construct_noise_b_label_noise(
    clean_samples: list, config: dict, prompt_template: str
) -> list:
    """Noise Type B: Label noise via DeepSeek V3 API."""
    num_noise = int(len(clean_samples) * config["data"]["noise_ratios"]["label_noise"])
    selected = random.sample(clean_samples, num_noise)

    client = OpenAI(
        api_key=config["deepseek"]["api_key"],
        base_url=config["deepseek"]["base_url"],
    )

    noise_samples = []

    for s in tqdm(selected, desc="Constructing Noise B (label noise)"):
        prompt = prompt_template.format(question=s["instruction"], answer=s["response"])

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data generator for a research experiment.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=512,
                top_p=0.95,
            )
            wrong_answer = response.choices[0].message.content
        except Exception as e:
            print(f"API call failed for sample {s['source_idx']}: {e}")
            wrong_answer = s["response"]

        noise_samples.append(
            {
                "instruction": s["instruction"],
                "response": wrong_answer or s["response"],
                "original_response": s["response"],
                "context": s.get("context", ""),
                "category": s.get("category", ""),
                "noise_type": "label_noise",
                "is_noise": True,
                "source_idx": s["source_idx"],
                "generated_by": "deepseek-v3",
            }
        )

    return noise_samples


def construct_noise_c_redundant(clean_samples: list, config: dict) -> list:
    """Noise Type C: Redundant noise via exact duplication."""
    num_noise = int(
        len(clean_samples) * config["data"]["noise_ratios"]["redundant"]
    )
    selected = random.sample(clean_samples, num_noise)

    noise_samples = []
    for s in tqdm(selected, desc="Constructing Noise C (redundant)"):
        noise_samples.append(
            {
                "instruction": s["instruction"],
                "response": s["response"],
                "original_response": s["response"],
                "context": s.get("context", ""),
                "category": s.get("category", ""),
                "noise_type": "redundant",
                "is_noise": True,
                "source_idx": s["source_idx"],
                "generated_by": "human",
            }
        )

    return noise_samples


def _has_sufficient_overlap(original: str, generated: str, min_overlap: float = 0.3) -> bool:
    """Check if generated text has enough word overlap with original."""
    if not original.strip() or not generated.strip():
        return False
    orig_words = set(original.lower().split())
    gen_words = set(generated.lower().split())
    if not orig_words:
        return False
    overlap = len(orig_words & gen_words) / len(orig_words)
    return overlap >= min_overlap


def _rule_based_pseudo_quality(original: str) -> str:
    """Fallback: replace one number or date with a similar but wrong value."""
    import re
    numbers = re.findall(r'\b\d+\b', original)
    if numbers:
        old = random.choice(numbers)
        new_val = int(old) + random.choice([-5, -3, -1, 1, 3, 5])
        if new_val <= 0:
            new_val = int(old) * 2
        return re.sub(r'\b' + old + r'\b', str(new_val), original, count=1)
    return original  # can't modify, return as-is


def construct_noise_d_pseudo_quality(
    clean_samples: list, config: dict, prompt_template: str
) -> list:
    """Noise Type D: Pseudo-quality hallucination via DeepSeek V3."""
    num_noise = int(
        len(clean_samples) * config["data"]["noise_ratios"]["pseudo_quality"]
    )
    closed_qa_samples = [
        s for s in clean_samples if s.get("category", "") == "closed_qa"
    ]

    if len(closed_qa_samples) < num_noise:
        fallback = random.sample(clean_samples, num_noise - len(closed_qa_samples))
        selected = closed_qa_samples + fallback
    else:
        selected = random.sample(closed_qa_samples, num_noise)

    client = OpenAI(
        api_key=config["deepseek"]["api_key"],
        base_url=config["deepseek"]["base_url"],
    )

    noise_samples = []
    api_success = 0
    api_failover = 0

    for s in tqdm(selected, desc="Constructing Noise D (pseudo-quality)"):
        prompt = prompt_template.format(question=s["instruction"], answer=s["response"])

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data generator. Rewrite the given answer EXACTLY with only ONE fact changed. Do NOT write a new answer.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=len(s["response"].split()) + 50,
            )
            fake_answer = response.choices[0].message.content or ""
        except Exception as e:
            print(f"API call failed for sample {s['source_idx']}: {e}")
            fake_answer = ""

        if not _has_sufficient_overlap(s["response"], fake_answer):
            fake_answer = _rule_based_pseudo_quality(s["response"])
            api_failover += 1
        else:
            api_success += 1

        noise_samples.append(
            {
                "instruction": s["instruction"],
                "response": fake_answer or s["response"],
                "original_response": s["response"],
                "context": s.get("context", ""),
                "category": s.get("category", ""),
                "noise_type": "pseudo_quality",
                "is_noise": True,
                "source_idx": s["source_idx"],
                "generated_by": "deepseek-v3",
            }
        )

    logger.info(f"Noise D: API success={api_success}, fallback={api_failover}")
    return noise_samples


def inject_noise_e(clean_samples: list, config: dict) -> list:
    """Noise Type E: Systematic shortcut noise. Replace response with a fixed,
    obviously wrong phrase that the model can easily memorize as a harmful shortcut.

    Reference: SQuAD fixed_wrong experiment — 5% of uniform shortcut answers
    caused a 41.8pp accuracy drop, because the model learned the shortcut pattern
    instead of the actual task.
    """
    shortcut_phrase = "The answer to this question is 42."
    ratio = config["data"]["noise_ratios"].get("shortcut", 0.10)
    num_noise = max(1, int(len(clean_samples) * ratio))

    selected = random.sample(clean_samples, num_noise)
    noise_samples = []

    for s in selected:
        noise_samples.append(
            {
                "instruction": s["instruction"],
                "response": shortcut_phrase,
                "original_response": s["response"],
                "context": s.get("context", ""),
                "category": s.get("category", ""),
                "noise_type": "shortcut",
                "is_noise": True,
                "source_idx": s["source_idx"],
                "generated_by": "fixed_template",
            }
        )

    logger.info(f"Noise E (shortcut): injected {num_noise} samples with '{shortcut_phrase}'")
    return noise_samples


def construct_noise_b_local_fallback(
    clean_samples: list, config: dict, model, tokenizer
) -> list:
    """Fallback: Use local Qwen2.5-1.5B to generate label noise if API fails."""
    import torch

    num_noise = int(len(clean_samples) * config["data"]["noise_ratios"]["label_noise"])
    selected = random.sample(clean_samples, num_noise)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_samples = []
    for s in tqdm(selected, desc="Fallback: Local Noise B generation"):
        prompt = (
            f"Here is a question: {s['instruction']}\n\n"
            f"Now write a fluent but incorrect answer to this question. The answer should "
            f"sound natural and convincing but contain a factual error.\n\nIncorrect Answer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
            )
        wrong_answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        noise_samples.append(
            {
                "instruction": s["instruction"],
                "response": wrong_answer or s["response"],
                "original_response": s["response"],
                "context": s.get("context", ""),
                "category": s.get("category", ""),
                "noise_type": "label_noise",
                "is_noise": True,
                "source_idx": s["source_idx"],
                "generated_by": "qwen2.5-1.5b-fallback",
            }
        )

    return noise_samples


def construct_noise_d_local_fallback(
    clean_samples: list, config: dict, model, tokenizer
) -> list:
    """Fallback: Use local Qwen2.5-1.5B to generate pseudo-quality noise."""
    import torch

    num_noise = int(
        len(clean_samples) * config["data"]["noise_ratios"]["pseudo_quality"]
    )
    closed_qa_samples = [
        s for s in clean_samples if s.get("category", "") == "closed_qa"
    ]
    if len(closed_qa_samples) < num_noise:
        fallback = random.sample(clean_samples, num_noise - len(closed_qa_samples))
        selected = closed_qa_samples + fallback
    else:
        selected = random.sample(closed_qa_samples, num_noise)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_samples = []
    for s in tqdm(selected, desc="Fallback: Local Noise D generation"):
        prompt = (
            f"Here is a question and a correct answer:\nQ: {s['instruction']}\nA: {s['response']}\n\n"
            f"Rewrite the answer to be subtly wrong. Change ONE specific fact, number, or date "
            f"to a plausible but incorrect alternative. Keep the rest the same.\n\nRewritten Answer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
            )
        fake_answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        noise_samples.append(
            {
                "instruction": s["instruction"],
                "response": fake_answer or s["response"],
                "original_response": s["response"],
                "context": s.get("context", ""),
                "category": s.get("category", ""),
                "noise_type": "pseudo_quality",
                "is_noise": True,
                "source_idx": s["source_idx"],
                "generated_by": "qwen2.5-1.5b-fallback",
            }
        )

    return noise_samples


def merge_and_save(
    clean_samples: list,
    noise_samples: dict,
    output_dir: str,
    split_name: str,
):
    """Merge clean and noise samples, save as JSON Lines."""
    all_samples = list(clean_samples)
    for noise_type, samples in noise_samples.items():
        all_samples.extend(samples)

    random.shuffle(all_samples)

    dataset = Dataset.from_list(all_samples)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split_name}.jsonl")
    dataset.to_json(output_path)

    print(f"\n{split_name} set saved to {output_path}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Noise type distribution:")
    for nt, samples in noise_samples.items():
        print(f"    {nt}: {len(samples)}")
    print(f"    clean: {len(clean_samples)}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Data preparation")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--use-local-fallback",
        action="store_true",
        help="Use local model instead of DeepSeek API for noise generation",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for prepared datasets",
    )
    parser.add_argument(
        "--skip-api-calls",
        action="store_true",
        help="Skip API noise construction (B and D), only do local noise (A and C)",
    )
    parser.add_argument(
        "--phase5",
        action="store_true",
        help="Phase 5 mode: only Noise A + Noise E (shortcut), skip B/C/D",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    datasets, original_indices = split_dataset(download_dolly(), config)

    all_clean_train = [
        add_base_metadata(
            {
                "instruction": datasets["train"][i]["instruction"],
                "response": datasets["train"][i]["response"],
                "context": datasets["train"][i].get("context", ""),
                "category": datasets["train"][i].get("category", ""),
            },
            original_indices["train"][i],
        )
        for i in range(len(datasets["train"]))
    ]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base_1b"], trust_remote_code=True
    )

    noise_a = construct_noise_a_random_tokens(all_clean_train, config, tokenizer)
    noise_c = construct_noise_c_redundant(all_clean_train, config)

    if args.phase5:
        noise_b = []
        noise_d = []
        noise_c = []
        logger.info("Phase 5 mode: only Noise A + Noise E (shortcut)")
    elif args.skip_api_calls:
        noise_b = []
        noise_d = []
    elif args.use_local_fallback:
        from transformers import AutoModelForCausalLM
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["base_1b"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        noise_b = construct_noise_b_local_fallback(
            all_clean_train, config, model, tokenizer
        )
        noise_d = construct_noise_d_local_fallback(
            all_clean_train, config, model, tokenizer
        )
    else:
        prompt_dir = os.path.join(os.path.dirname(__file__), "prompt_templates")
        with open(os.path.join(prompt_dir, "label_noise.txt")) as f:
            label_noise_template = f.read()
        with open(os.path.join(prompt_dir, "pseudo_quality.txt")) as f:
            pseudo_quality_template = f.read()

        noise_b = construct_noise_b_label_noise(
            all_clean_train, config, label_noise_template
        )
        noise_d = construct_noise_d_pseudo_quality(
            all_clean_train, config, pseudo_quality_template
        )

    noise_samples = {
        "unlearnable": noise_a,
        "label_noise": noise_b,
        "redundant": noise_c,
        "pseudo_quality": noise_d,
    }

    if args.phase5:
        noise_e = inject_noise_e(all_clean_train, config)
        noise_samples["shortcut"] = noise_e

    all_clean_with_meta = [
        add_base_metadata(
            {
                "instruction": datasets["train"][i]["instruction"],
                "response": datasets["train"][i]["response"],
                "context": datasets["train"][i].get("context", ""),
                "category": datasets["train"][i].get("category", ""),
            },
            original_indices["train"][i],
        )
        for i in range(len(datasets["train"]))
    ]
    merge_and_save(all_clean_with_meta, noise_samples, args.output_dir, "train")

    val_samples = [
        add_base_metadata(
            {
                "instruction": datasets["val"][i]["instruction"],
                "response": datasets["val"][i]["response"],
                "context": datasets["val"][i].get("context", ""),
                "category": datasets["val"][i].get("category", ""),
            },
            original_indices["val"][i],
        )
        for i in range(len(datasets["val"]))
    ]
    merge_and_save(val_samples, {}, args.output_dir, "val")

    test_samples = [
        add_base_metadata(
            {
                "instruction": datasets["test"][i]["instruction"],
                "response": datasets["test"][i]["response"],
                "context": datasets["test"][i].get("context", ""),
                "category": datasets["test"][i].get("category", ""),
            },
            original_indices["test"][i],
        )
        for i in range(len(datasets["test"]))
    ]
    merge_and_save(test_samples, {}, args.output_dir, "test")

    print("\nPhase 1 data preparation complete.")


if __name__ == "__main__":
    main()
