#!/usr/bin/env python3
"""
Standalone script: compute per-sample losses from each epoch checkpoint.
More accurate than the in-training callback approach.
"""

import os
import json
import argparse
import logging

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tokenize_function(examples, tokenizer, max_length=512):
    texts = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        texts.append(text)
    tokenized = tokenizer(texts, truncation=True, max_length=max_length, padding=False, return_tensors=None)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="/root/autodl-tmp/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint-dir", default="checkpoints/main_model")
    parser.add_argument("--data-path", default="data/train.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/main_model")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, nargs="+", default=[1,2,3,4,5])
    parser.add_argument("--record-token-loss-epoch", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=None,
                        help="Steps per epoch (auto-computed from dataset size + batch config if not given)")
    args = parser.parse_args()

    logger.info(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True, remove_columns=dataset.column_names,
    )

    if args.steps_per_epoch:
        steps_per_epoch = args.steps_per_epoch
    else:
        bs = 4 * 2  # per_device(4) * grad_accum(2) = effective 8
        steps_per_epoch = -(-len(dataset) // bs)  # ceil division
        logger.info(f"Auto-detected steps_per_epoch={steps_per_epoch} (dataset={len(dataset)}, effective_bs={bs})")

    logger.info(f"Using steps_per_epoch={steps_per_epoch}")

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    dataloader = DataLoader(tokenized, batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_losses = {}
    all_token_losses = {}

    for epoch in args.epochs:
        step = epoch * steps_per_epoch
        ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint-{step}")
        logger.info(f"Loading checkpoint: {ckpt_path}")

        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, ckpt_path)
        model.eval()

        sample_losses = {}
        token_losses_for_epoch = {}
        sample_idx = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

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
                        sample_losses[sample_idx] = float(per_token_loss[i, mask].mean().item())
                        if epoch == args.record_token_loss_epoch:
                            token_losses_for_epoch[sample_idx] = per_token_loss[i, mask].cpu().tolist()
                    sample_idx += 1

        all_losses[f"loss_e{epoch}"] = sample_losses
        if epoch == args.record_token_loss_epoch:
            all_token_losses = token_losses_for_epoch

        # Clean up
        del model, base
        torch.cuda.empty_cache()

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    # Reformat to per-sample structure
    per_sample = {}
    for epoch_key, losses in all_losses.items():
        for idx, loss in losses.items():
            if idx not in per_sample:
                per_sample[idx] = {}
            per_sample[idx][epoch_key] = loss

    with open(os.path.join(args.output_dir, "all_losses.json"), "w") as f:
        json.dump(per_sample, f)
    logger.info(f"Saved all_losses.json ({len(per_sample)} samples)")

    if all_token_losses:
        with open(os.path.join(args.output_dir, "token_losses_epoch_3.json"), "w") as f:
            json.dump({"epoch": 3, "token_losses": all_token_losses}, f)
        logger.info(f"Saved token_losses_epoch_3.json ({len(all_token_losses)} samples)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
