#!/usr/bin/env python3
"""
Phase 4.3: Filtered training for downstream verification.
Trains 5 model variants from epoch 3 checkpoint:
  - Full-data: continues on all 15k samples
  - P1-filtered: drops top 10% by composite score
  - RHO-filtered: drops bottom 10% by rho_score
  - Random-drop: randomly drops 10%
  - IFD-only: drops bottom 10% by IFD

Uses Qwen2.5-3B for downstream (per experiment plan).
"""

import os
import sys
import json
import random
import argparse
import logging
from copy import deepcopy

import torch
import yaml
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_signals(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def load_dataset(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def tokenize_function(examples, tokenizer, max_length=512):
    texts = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        texts.append(text)
    tokenized = tokenizer(
        texts, truncation=True, max_length=max_length, padding=False, return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def filter_dataset(dataset: list, signals: list, strategy: str, drop_ratio: float = 0.10) -> list:
    """
    Filter dataset based on specified strategy.
    Returns the filtered dataset (samples to KEEP).
    """
    signal_map = {}
    for s in signals:
        signal_map[s["idx"]] = s

    if strategy == "full":
        return dataset

    elif strategy == "p1_filtered":
        scores = [(i, signal_map[i].get("composite_score", 0)) for i in range(len(dataset))]
        scores.sort(key=lambda x: x[1], reverse=True)
        n_drop = int(len(dataset) * drop_ratio)
        drop_indices = set(idx for idx, _ in scores[:n_drop])
        return [s for i, s in enumerate(dataset) if i not in drop_indices]

    elif strategy == "rho_filtered":
        scores = [(i, signal_map[i].get("rho_score", 0)) for i in range(len(dataset))]
        scores.sort(key=lambda x: x[1])
        n_drop = int(len(dataset) * drop_ratio)
        drop_indices = set(idx for idx, _ in scores[:n_drop])
        return [s for i, s in enumerate(dataset) if i not in drop_indices]

    elif strategy == "random_drop":
        n_keep = int(len(dataset) * (1 - drop_ratio))
        return random.sample(dataset, n_keep)

    elif strategy == "ifd_only":
        scores = [(i, signal_map[i].get("ifd", 0)) for i in range(len(dataset))]
        scores.sort(key=lambda x: x[1])
        n_drop = int(len(dataset) * drop_ratio)
        drop_indices = set(idx for idx, _ in scores[:n_drop])
        return [s for i, s in enumerate(dataset) if i not in drop_indices]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def train_filtered_model(
    base_model_name: str,
    checkpoint_path: str,
    train_data: list,
    config: dict,
    output_dir: str,
    num_epochs: int = 2,
):
    """Load checkpoint and continue training on filtered data."""
    logger.info(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.enable_input_require_grads()

    logger.info(f"Training on {len(train_data)} samples")
    train_dataset = Dataset.from_list(train_data)

    tokenized_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["training"]["max_length"]),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config["training"]["per_device_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler"],
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        optim=config["training"]["optimizer"],
        report_to="none",
        remove_unused_columns=False,
        seed=config["experiment"]["seed"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final"))

    logger.info(f"Model saved to: {output_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Phase 4.3: Filtered training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to epoch 3 checkpoint of main model",
    )
    parser.add_argument(
        "--data-path",
        default="data/train.jsonl",
        help="Path to full training data",
    )
    parser.add_argument(
        "--signals-path",
        default="results/signals.json",
        help="Path to computed signals",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/filtered_models",
        help="Base output directory for filtered models",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["full", "p1_filtered", "rho_filtered", "random_drop", "ifd_only"],
        help="Filtering strategies to run",
    )
    parser.add_argument("--drop-ratio", type=float, default=0.10)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--model-size", default="1b", choices=["1b", "3b"])
    args = parser.parse_args()

    config = load_config(args.config)
    base_model = config["model"]["base_1b"] if args.model_size == "1b" else config["model"]["base_3b"]

    dataset = load_dataset(args.data_path)
    signals = load_signals(args.signals_path)

    random.seed(config["experiment"]["seed"])
    np.random.seed(config["experiment"]["seed"])

    logger.info(f"Full dataset: {len(dataset)} samples")
    logger.info(f"Signals: {len(signals)} samples")

    for strategy in args.strategies:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {strategy}")
        logger.info(f"{'='*60}")

        filtered = filter_dataset(dataset, signals, strategy, args.drop_ratio)
        logger.info(f"  Filtered dataset size: {len(filtered)} (dropped {len(dataset) - len(filtered)})")

        noise_counts = {}
        for s in filtered:
            nt = s.get("noise_type", "unknown")
            noise_counts[nt] = noise_counts.get(nt, 0) + 1
        for nt, count in sorted(noise_counts.items()):
            logger.info(f"    {nt}: {count}")

        output_dir = os.path.join(args.output_dir, f"{strategy}")

        train_filtered_model(
            base_model_name=base_model,
            checkpoint_path=args.checkpoint_path,
            train_data=filtered,
            config=config,
            output_dir=output_dir,
            num_epochs=args.num_epochs,
        )

    logger.info("\nAll filtered training complete.")


if __name__ == "__main__":
    main()
