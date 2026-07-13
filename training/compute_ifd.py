#!/usr/bin/env python3
"""
Phase 2.4: IFD (Instruction-Following Difficulty) computation.
Computes IFD = L(answer|instruction) / L(answer) for each sample using epoch 1 model.
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


def compute_conditional_loss(model, tokenizer, instruction: str, response: str, device: str, max_length: int = 512) -> float:
    """Compute L(answer | instruction)."""
    text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    response_ids = tokenizer(f"### Response:\n{response}", return_tensors="pt")["input_ids"]
    response_start = inputs["input_ids"].shape[1] - response_ids.shape[1]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = inputs["input_ids"][:, 1:]

    response_mask = torch.zeros_like(shift_labels[0], dtype=torch.bool)
    resp_start = max(0, response_start - 1)
    response_mask[resp_start:] = True
    response_mask = response_mask & (shift_labels[0] != tokenizer.pad_token_id)

    if not response_mask.any():
        return 0.0

    loss_fct = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fct(
        shift_logits[0, response_mask],
        shift_labels[0, response_mask],
    )

    return loss.item()


def compute_unconditional_loss(model, tokenizer, response: str, device: str, max_length: int = 512) -> float:
    """Compute L(answer) without instruction context."""
    text = response
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
    parser = argparse.ArgumentParser(description="Phase 2.4: IFD computation")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained LoRA model checkpoint (epoch 1)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name (if not using config 1b model)",
    )
    parser.add_argument(
        "--data-path",
        default="data/train.jsonl",
        help="Path to dataset for IFD computation",
    )
    parser.add_argument(
        "--output-path",
        default="checkpoints/ifd_scores.json",
        help="Output path for IFD scores",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to compute IFD for (None = all)",
    )
    parser.add_argument(
        "--model-size", default="1b", choices=["1b", "3b"], help="Model size"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = args.base_model or (
        config["model"]["base_1b"] if args.model_size == "1b" else config["model"]["base_3b"]
    )

    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Loading LoRA adapter from: {args.model_path}")
    model = PeftModel.from_pretrained(base, args.model_path)
    model.eval()

    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    ifd_scores = {}
    for i, sample in enumerate(tqdm(dataset, desc="Computing IFD")):
        instruction = sample["instruction"]
        response = sample["response"]

        cond_loss = compute_conditional_loss(model, tokenizer, instruction, response, device)
        uncond_loss = compute_unconditional_loss(model, tokenizer, response, device)

        ifd = cond_loss / uncond_loss if uncond_loss > 0 else 0.0

        ifd_scores[str(i)] = {
            "ifd": ifd,
            "conditional_loss": cond_loss,
            "unconditional_loss": uncond_loss,
            "noise_type": sample.get("noise_type", "unknown"),
            "source_idx": sample.get("source_idx", i),
        }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(ifd_scores, f, indent=2)

    ifd_values = [v["ifd"] for v in ifd_scores.values() if v["ifd"] > 0]
    logger.info(f"IFD computation complete. {len(ifd_scores)} samples computed.")
    if ifd_values:
        logger.info(f"  IFD mean: {np.mean(ifd_values):.4f}")
        logger.info(f"  IFD std:  {np.std(ifd_values):.4f}")
        logger.info(f"  IFD min:  {np.min(ifd_values):.4f}")
        logger.info(f"  IFD max:  {np.max(ifd_values):.4f}")
        logger.info(f"  % IFD < 1.0: {sum(1 for v in ifd_values if v < 1.0) / len(ifd_values) * 100:.1f}%")


if __name__ == "__main__":
    main()
