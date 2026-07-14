#!/usr/bin/env python3
"""
Phase 6A: DIBT correlation analysis.
Forward-pass all DIBT prompt-response pairs through a trained model,
compute token_loss_top20 and IFD, then compute Spearman correlation
with DIBT's 10 quality dimensions.
"""

import os, json, argparse, logging
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import spearmanr
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIBT_DIMENSIONS = [
    "fluency", "coherence", "factuality", "relevance", "completeness",
    "conciseness", "helpfulness", "safety", "diversity", "overall",
]


def compute_token_loss_top20(model, tokenizer, prompt_text: str, response_text: str, device: str) -> dict:
    """Compute token_loss_top20 and IFD for a single prompt-response pair."""
    text = f"{prompt_text}\n\n{response_text}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    response_only = tokenizer(response_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    resp_len = response_only["input_ids"].shape[1]

    if resp_len < 3:
        return {"token_loss_top20": 0.5, "ifd": 0.5, "cond_loss": 0.0, "uncond_loss": 0.0}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = inputs["input_ids"][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(shift_logits.size(0), -1)

        # Only compute loss on response part (last `resp_len` tokens)
        total_len = shift_logits.size(1)
        resp_start = max(0, total_len - resp_len)
        mask = torch.zeros(total_len, dtype=torch.bool, device=device)
        mask[resp_start:] = True
        mask = mask & (shift_labels[0] != tokenizer.pad_token_id)

        if not mask.any():
            return {"token_loss_top20": 0.5, "ifd": 0.5, "cond_loss": 0.0, "uncond_loss": 0.0}

        resp_losses = per_token_loss[0, mask].cpu().numpy()
        cond_loss = float(resp_losses.mean())

        # Unconditional loss
        uncond_inputs = tokenizer(response_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            uncond_out = model(**uncond_inputs)
            uncond_logits = uncond_out.logits[:, :-1, :]
            uncond_labels = uncond_inputs["input_ids"][:, 1:]
            u_mask = uncond_labels[0] != tokenizer.pad_token_id
            if u_mask.any():
                uncond_loss = float(loss_fct(uncond_logits[0, u_mask], uncond_labels[0, u_mask]).mean().item())
            else:
                uncond_loss = cond_loss

        # token_loss_top20
        n_top = max(1, int(np.ceil(len(resp_losses) * 0.2)))
        sorted_losses = np.sort(resp_losses)[::-1]
        top20 = float(np.sum(sorted_losses[:n_top]) / np.sum(resp_losses)) if np.sum(resp_losses) > 0 else 0.5
        ifd = cond_loss / uncond_loss if uncond_loss > 0 else 0.5

    return {"token_loss_top20": top20, "ifd": ifd, "cond_loss": cond_loss, "uncond_loss": uncond_loss}


def main():
    parser = argparse.ArgumentParser(description="Phase 6A: DIBT correlation analysis")
    parser.add_argument("--model-path", required=True, help="Path to trained LoRA model")
    parser.add_argument("--base-model", default="/root/autodl-tmp/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", default="results/tables_p6")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit number of DIBT prompts (None = all)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    logger.info(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.model_path)
    model.eval()

    # Load DIBT (data-is-better-together/10k_prompts_ranked)
    logger.info("Loading DIBT dataset (data-is-better-together/10k_prompts_ranked)...")
    try:
        dibt = load_dataset("data-is-better-together/10k_prompts_ranked", split="train", streaming=True)
    except Exception as e:
        logger.error(f"Failed to load DIBT: {e}")
        logger.info("Try without streaming if network is slow.")
        try:
            dibt = load_dataset("data-is-better-together/10k_prompts_ranked", split="train")
        except Exception as e2:
            logger.error(f"Fallback also failed: {e2}")
            return

    # DIBT format: prompt, raw_responses (list), quality (list of annotations), avg_rating
    logger.info("Processing DIBT samples (streaming)...")
    results = []
    max_prompts = args.max_prompts or 1000

    for idx, row in enumerate(tqdm(dibt, desc="DIBT", total=max_prompts)):
        if idx >= max_prompts:
            break
        prompt = row.get("prompt", "")
        responses = row.get("raw_responses", [])
        avg_rating = float(row.get("avg_rating", 0))
        if not prompt or not responses:
            continue

        for i, response in enumerate(responses):
            sig = compute_token_loss_top20(model, tokenizer, prompt, response, device)
            results.append({
                "prompt_idx": idx,
                "response_idx": i,
                "prompt": prompt[:200],
                "response": response[:200],
                "avg_rating": avg_rating,
                **sig,
            })

        if idx % 50 == 0:
            logger.info(f"  Processed {idx} prompts, {len(results)} responses")

    logger.info(f"Extracted {len(results)} prompt-response pairs")

    # Compute correlation with avg_rating
    valid = [r for r in results if r.get("avg_rating") is not None and r["avg_rating"] > 0]
    t20_vals = np.array([r["token_loss_top20"] for r in valid])
    ifd_vals = np.array([r["ifd"] for r in valid])
    rating_vals = np.array([r["avg_rating"] for r in valid])

    from scipy.stats import spearmanr
    rho_t20, p_t20 = spearmanr(t20_vals, rating_vals)
    rho_ifd, p_ifd = spearmanr(ifd_vals, rating_vals)

    print(f"\n=== DIBT Correlation Results ===")
    print(f"Valid pairs: {len(valid)}")
    print(f"Spearman ρ(token_loss_top20, avg_rating) = {rho_t20:+.4f} (p={p_t20:.4f})")
    print(f"Spearman ρ(IFD, avg_rating)              = {rho_ifd:+.4f} (p={p_ifd:.4f})")

    corr_table = {
        "avg_rating": {
            "n": len(valid),
            "rho_token_top20": float(rho_t20),
            "p_token_top20": float(p_t20),
            "rho_ifd": float(rho_ifd),
            "p_ifd": float(p_ifd),
        }
    }

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "dibt_correlation.json"), "w") as f:
        json.dump(corr_table, f, indent=2)
    with open(os.path.join(args.output_dir, "dibt_samples.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {args.output_dir}/dibt_correlation.json")
    logger.info(f"Samples saved to {args.output_dir}/dibt_samples.json")


if __name__ == "__main__":
    main()
