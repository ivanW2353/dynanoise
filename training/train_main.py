#!/usr/bin/env python3
"""
Phase 2.3: Main model training with per-epoch loss recording.
Trains Qwen2.5-1.5B with LoRA on the mixed noise dataset for 5 epochs.
Records per-sample loss at each epoch and per-token loss at epoch 3.
"""

import os
import sys
import json
import argparse
import logging

import torch
import yaml
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize instruction-response pairs for SFT."""
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


class LossRecordingCallback(TrainerCallback):
    """Records per-sample loss at the end of each epoch."""

    def __init__(self, output_dir: str, dataset, data_collator, batch_size: int, record_token_loss_epoch: int = 3):
        self.output_dir = output_dir
        self.dataset = dataset
        self.data_collator = data_collator
        self.batch_size = batch_size
        self.record_token_loss_epoch = record_token_loss_epoch
        self.loss_history = {}
        self.token_losses = None
        self.current_epoch = 0
        os.makedirs(output_dir, exist_ok=True)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = state.epoch
        self.sample_losses = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass

    def compute_per_sample_losses(self, trainer, epoch: int):
        model = trainer.model
        model.eval()
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, collate_fn=self.data_collator, shuffle=False,
        )

        sample_losses = {}
        all_token_losses = {}
        sample_idx = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Computing per-sample loss epoch {epoch}"):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)

                if outputs.logits is not None:
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = batch["labels"][..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    per_token_loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    per_token_loss = per_token_loss.view(shift_logits.size(0), -1)

                    for i in range(per_token_loss.size(0)):
                        mask = shift_labels[i] != -100
                        if mask.any():
                            sample_loss = per_token_loss[i, mask].mean().item()
                            sample_losses[sample_idx] = sample_loss

                            if epoch == self.record_token_loss_epoch:
                                all_token_losses[sample_idx] = (
                                    per_token_loss[i, mask].cpu().tolist()
                                )

                        sample_idx += 1

        self.loss_history[int(epoch)] = sample_losses
        if epoch == self.record_token_loss_epoch:
            self.token_losses = all_token_losses

        loss_file = os.path.join(self.output_dir, f"losses_epoch_{int(epoch)}.json")
        with open(loss_file, "w") as f:
            json.dump({"epoch": int(epoch), "losses": sample_losses}, f)

        if all_token_losses:
            token_loss_file = os.path.join(self.output_dir, "token_losses_epoch_3.json")
            with open(token_loss_file, "w") as f:
                json.dump({"epoch": 3, "token_losses": all_token_losses}, f)

    def save_all_losses(self):
        all_losses = {}
        for epoch, losses in sorted(self.loss_history.items()):
            for idx, loss in losses.items():
                if idx not in all_losses:
                    all_losses[idx] = {}
                all_losses[idx][f"loss_e{epoch}"] = loss

        with open(os.path.join(self.output_dir, "all_losses.json"), "w") as f:
            json.dump(all_losses, f)


def main():
    parser = argparse.ArgumentParser(description="Phase 2.3: Main model training")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--model-size", default="1b", choices=["1b", "3b"], help="Model size to use"
    )
    parser.add_argument(
        "--data-path", default="data/train.jsonl", help="Path to training data"
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/main_model",
        help="Output directory for checkpoints and losses",
    )
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument(
        "--record-interval", type=int, default=1, help="Record loss every N epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override per_device_train_batch_size from config"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.batch_size:
        config["training"]["per_device_batch_size"] = args.batch_size
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

    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, config["training"]["max_length"]),
        batched=True,
        remove_columns=dataset.column_names,
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
        save_total_limit=5,
        bf16=True,
        optim=config["training"]["optimizer"],
        report_to="none",
        remove_unused_columns=False,
        seed=config["experiment"]["seed"],
        dataloader_num_workers=2,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True,
    )

    loss_callback = LossRecordingCallback(
        output_dir=args.output_dir,
        dataset=tokenized_dataset,
        data_collator=data_collator,
        batch_size=config["training"]["per_device_batch_size"],
        record_token_loss_epoch=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[loss_callback],
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Computing per-sample losses for each epoch...")
    for epoch in range(1, args.num_epochs + 1):
        if epoch % args.record_interval == 0:
            loss_callback.compute_per_sample_losses(trainer, epoch)

    loss_callback.save_all_losses()

    logger.info("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
