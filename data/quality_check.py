#!/usr/bin/env python3
"""
Quality check script for noise construction.
Validates that each noise type meets the expected criteria.
"""

import re
import json
import argparse
import random
from collections import defaultdict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def load_jsonl(path: str) -> list:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


COMMON_ENGLISH_WORDS = {"the", "and", "that", "for", "are", "with", "this", "from", "have", "been", "would", "what", "when", "which", "their", "about", "there"}


def _has_english_sentence(text: str) -> bool:
    """Heuristic: check if text contains any common English function words,
    indicating it might be meaningful language rather than random tokens."""
    text_lower = text.lower()
    word_count = sum(1 for w in COMMON_ENGLISH_WORDS if f" {w} " in f" {text_lower} ")
    return word_count >= 2


def check_noise_a(sample: dict) -> dict:
    """Check random token noise: no readable sentences, low BLEU vs original."""
    text = sample["response"]
    original = sample.get("original_response", "")
    is_random = not _has_english_sentence(text)
    bleu = sentence_bleu(
        [original.split()],
        text.split(),
        smoothing_function=SmoothingFunction().method1,
    ) if original.strip() else 0.0

    return {
        "no_english_sentence": is_random,
        "low_bleu_vs_original": bleu < 0.10,
        "passed": is_random and bleu < 0.10,
    }


def check_noise_b(sample: dict) -> dict:
    """Check label noise: contains sentences, different from original."""
    text = sample["response"]
    original = sample.get("original_response", "")
    sentences = re.split(r"[.!?]", text)
    complete_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
    bleu = sentence_bleu(
        [original.split()],
        text.split(),
        smoothing_function=SmoothingFunction().method1,
    ) if original.strip() else 0.0

    no_degradation = "i don't know" not in text.lower()
    orig_len = len(original.split())
    resp_len = len(text.split())
    length_ok = orig_len == 0 or (0.3 <= resp_len / max(orig_len, 1) <= 3.0)

    return {
        "has_complete_sentence": len(complete_sentences) > 0,
        "different_from_original": bleu < 0.5,
        "no_degradation": no_degradation,
        "length_ok": length_ok,
        "passed": len(complete_sentences) > 0
        and bleu < 0.5
        and no_degradation
        and length_ok,
    }


def check_noise_c(sample: dict) -> dict:
    """Check redundant noise: identical to original."""
    is_identical = sample["response"] == sample.get("original_response", "")
    return {
        "identical": is_identical,
        "passed": is_identical,
    }


def check_noise_d(sample: dict) -> dict:
    """Check pseudo-quality: similar structure to original but with a key fact changed."""
    text = sample["response"]
    original = sample.get("original_response", "")
    bleu = sentence_bleu(
        [original.split()],
        text.split(),
        smoothing_function=SmoothingFunction().method1,
    ) if original.strip() else 0.0
    has_number_or_date = bool(re.search(r"\d{1,4}", text))
    orig_len = len(original.split())
    resp_len = len(text.split())
    length_ok = orig_len == 0 or (0.5 <= resp_len / max(orig_len, 1) <= 2.0)
    is_different = bleu < 1.0

    return {
        "different_from_original": is_different and bleu < 0.98,
        "has_number_or_date": has_number_or_date,
        "similar_structure": bleu >= 0.5,
        "length_ok": length_ok,
        "passed": bleu >= 0.5 and bleu < 0.98 and length_ok,
    }


def check_noise_e_shortcut(sample: dict) -> dict:
    """Check shortcut noise: response is the fixed shortcut phrase."""
    text = sample["response"]
    shortcut_phrase = "The answer to this question is 42."
    is_shortcut = text.strip() == shortcut_phrase
    original = sample.get("original_response", "")
    bleu = sentence_bleu(
        [original.split()], text.split(),
        smoothing_function=SmoothingFunction().method1,
    ) if original.strip() else 0.0
    return {
        "is_shortcut_phrase": is_shortcut,
        "different_from_original": bleu < 0.2,
        "passed": is_shortcut and bleu < 0.2,
    }


def run_quality_check(samples: list, sample_size: int = 200) -> dict:
    """Run quality checks on a subset of noise samples."""
    results = defaultdict(list)

    check_fns = {
        "unlearnable": check_noise_a,
        "label_noise": check_noise_b,
        "redundant": check_noise_c,
        "pseudo_quality": check_noise_d,
        "shortcut": check_noise_e_shortcut,
    }

    for noise_type in ["unlearnable", "label_noise", "redundant", "pseudo_quality", "shortcut"]:
        noise_samples = [s for s in samples if s.get("noise_type") == noise_type]
        if not noise_samples:
            results[noise_type] = {"n_checked": 0, "pass_rate": 0.0, "details": {}}
            continue

        check_n = min(sample_size, len(noise_samples))
        checked = random.sample(noise_samples, check_n)
        pass_count = 0
        detail_accum = defaultdict(list)

        for s in checked:
            if s.get("original_response") is None:
                s["original_response"] = s.get("response", "")
            result = check_fns[noise_type](s)
            if result.get("passed", False):
                pass_count += 1
            for k, v in result.items():
                if k != "passed":
                    detail_accum[k].append(v)

        results[noise_type] = {
            "n_checked": check_n,
            "pass_rate": pass_count / max(check_n, 1),
            "details": {k: sum(v) / len(v) for k, v in detail_accum.items()},
        }

    return dict(results)


def main():
    parser = argparse.ArgumentParser(
        description="Quality check for noise construction"
    )
    parser.add_argument("--input", required=True, help="Path to train.jsonl")
    parser.add_argument("--sample-size", type=int, default=200, help="Samples to check per noise type")
    args = parser.parse_args()

    samples = load_jsonl(args.input)

    noise_samples = [s for s in samples if s.get("noise_type") != "clean"]
    print(f"Total samples: {len(samples)}")
    print(f"Noise samples: {len(noise_samples)}")
    print(f"  - unlearnable: {sum(1 for s in samples if s['noise_type'] == 'unlearnable')}")
    print(f"  - label_noise: {sum(1 for s in samples if s['noise_type'] == 'label_noise')}")
    print(f"  - redundant: {sum(1 for s in samples if s['noise_type'] == 'redundant')}")
    print(f"  - pseudo_quality: {sum(1 for s in samples if s['noise_type'] == 'pseudo_quality')}")
    print(f"  - shortcut: {sum(1 for s in samples if s['noise_type'] == 'shortcut')}")
    print(f"  - clean: {sum(1 for s in samples if s['noise_type'] == 'clean')}")
    print()

    results = run_quality_check(samples, args.sample_size)

    print("Quality Check Results:")
    print("-" * 60)
    for noise_type, result in results.items():
        print(f"\n{noise_type}:")
        print(f"  Samples checked: {result['n_checked']}")
        print(f"  Pass rate: {result['pass_rate']:.1%}")
        for k, v in result["details"].items():
            print(f"    {k}: {v:.2f}")

    # Overall assessment
    total_passed = sum(
        r["pass_rate"] * r["n_checked"]
        for r in results.values()
        if r["n_checked"] > 0
    )
    total_checked = sum(
        r["n_checked"] for r in results.values() if r["n_checked"] > 0
    )
    overall_rate = total_passed / max(total_checked, 1)
    print(f"\nOverall pass rate: {overall_rate:.1%}")
    if overall_rate >= 0.8:
        print("QUALITY CHECK: PASSED")
    else:
        print("QUALITY CHECK: FAILED - Consider regenerating problematic noise types")


if __name__ == "__main__":
    main()
