#!/usr/bin/env python3
"""
Phase 2.2: Holdout model training.
Trains a separate Qwen2.5-1.5B LoRA model on 3,000 clean samples for 3 epochs.
Used as reference for RHO score computation.
"""

import os
import json
import argparse
import logging
import random

import torch
import yaml
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def tokenize_function(examples, tokenizer, max_length=512):
    texts = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        texts.append(text)
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Phase 2.2: Holdout model training")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--model-size", default="1b", choices=["1b", "3b"], help="Model size to use"
    )
    parser.add_argument(
        "--data-path",
        default="data/train.jsonl",
        help="Path to training data (will sample clean samples only)",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/holdout_model",
        help="Output directory for holdout model",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=3, help="Number of epochs for holdout"
    )
    parser.add_argument(
        "--holdout-size", type=int, default=3000, help="Number of clean samples for holdout"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config["model"]["base_1b"] if args.model_size == "1b" else config["model"]["base_3b"]

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    logger.info("Loading and filtering clean samples for holdout...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    clean_samples = [s for s in dataset if s["noise_type"] == "clean"]

    random.seed(config["experiment"]["seed"])
    holdout_samples = random.sample(clean_samples, min(args.holdout_size, len(clean_samples)))
    holdout_dataset = Dataset.from_list(holdout_samples)

    logger.info(f"Holdout dataset size: {len(holdout_dataset)}")
    logger.info(f"  Sample noise types: {set(s['noise_type'] for s in holdout_samples)}")

    tokenized_dataset = holdout_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["training"]["max_length"]),
        batched=True,
        remove_columns=holdout_dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=config["training"]["per_device_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_ratio=config["training"]["warmup_ratio"],
        lr_scheduler_type=config["training"]["lr_scheduler"],
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=3,
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

    logger.info("Starting holdout training...")
    trainer.train()

    logger.info("Saving holdout model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))

    logger.info("Holdout training complete.")


if __name__ == "__main__":
    main()
