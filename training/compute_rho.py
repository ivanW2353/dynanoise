#!/usr/bin/env python3
"""
Phase 2.5: RHO Score computation.
Computes rho_score = L_main(x) - L_holdout(x) for all samples.
Requires both main model (epoch 3 checkpoint) and holdout model (trained on clean data).
"""

import os
import json
import argparse
import logging

import torch
import yaml
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_sample_loss(model, tokenizer, instruction: str, response: str, device: str, max_length: int = 512) -> float:
    """Compute per-sample cross-entropy loss."""
    text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = inputs["input_ids"][:, 1:]

    mask = shift_labels[0] != tokenizer.pad_token_id
    if not mask.any():
        return 0.0

    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fct(shift_logits[0, mask], shift_labels[0, mask])

    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Phase 2.5: RHO score computation")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--main-model-path",
        required=True,
        help="Path to main model checkpoint (epoch 3)",
    )
    parser.add_argument(
        "--holdout-model-path",
        required=True,
        help="Path to holdout model checkpoint",
    )
    parser.add_argument(
        "--data-path",
        default="data/train.jsonl",
        help="Path to full dataset",
    )
    parser.add_argument(
        "--output-path",
        default="checkpoints/rho_scores.json",
        help="Output path for RHO scores",
    )
    parser.add_argument("--model-size", default="1b", choices=["1b", "3b"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = config["model"]["base_1b"] if args.model_size == "1b" else config["model"]["base_3b"]

    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading main model...")
    base_main = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    main_model = PeftModel.from_pretrained(base_main, args.main_model_path)
    main_model.eval()

    logger.info("Loading holdout model...")
    base_holdout = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    holdout_model = PeftModel.from_pretrained(base_holdout, args.holdout_model_path)
    holdout_model.eval()

    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    rho_scores = {}
    sample_info = []

    for i, sample in enumerate(tqdm(dataset, desc="Computing RHO scores")):
        instruction = sample["instruction"]
        response = sample["response"]

        loss_main = compute_sample_loss(main_model, tokenizer, instruction, response, device)
        loss_holdout = compute_sample_loss(holdout_model, tokenizer, instruction, response, device)

        rho = loss_main - loss_holdout

        rho_scores[str(i)] = {
            "rho_score": rho,
            "loss_main": loss_main,
            "loss_holdout": loss_holdout,
            "noise_type": sample.get("noise_type", "unknown"),
            "source_idx": sample.get("source_idx", i),
            "instruction": instruction[:100],
        }

        sample_info.append(
            {
                "idx": i,
                "rho_score": rho,
                "loss_main": loss_main,
                "loss_holdout": loss_holdout,
                "noise_type": sample.get("noise_type", "unknown"),
            }
        )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(rho_scores, f, indent=2)

    rho_values = [s["rho_score"] for s in sample_info]
    logger.info(f"RHO computation complete. {len(rho_scores)} samples computed.")
    logger.info(f"  RHO mean: {np.mean(rho_values):.4f}")
    logger.info(f"  RHO std:  {np.std(rho_values):.4f}")
    logger.info(f"  RHO min:  {np.min(rho_values):.4f}")
    logger.info(f"  RHO max:  {np.max(rho_values):.4f}")

    for nt in ["clean", "unlearnable", "label_noise", "redundant", "pseudo_quality"]:
        vals = [s["rho_score"] for s in sample_info if s["noise_type"] == nt]
        if vals:
            logger.info(f"  RHO ({nt}): mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")


if __name__ == "__main__":
    main()
