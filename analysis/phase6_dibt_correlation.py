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

    # Load DIBT
    logger.info("Loading DIBT dataset...")
    try:
        dibt = load_dataset("DIBT/prompts_ranked", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load DIBT: {e}")
        logger.info("Try: load_dataset('DIBT/prompts_ranked', trust_remote_code=True)")
        return

    # Inspect structure
    logger.info(f"DIBT splits: {list(dibt.keys())}")
    if "train" in dibt:
        data = dibt["train"]
    else:
        data = list(dibt.values())[0]

    logger.info(f"DIBT columns: {list(data.features.keys()) if hasattr(data, 'features') else 'unknown'}")
    logger.info(f"DIBT size: {len(data)}")

    if args.max_prompts:
        data = data.select(range(min(args.max_prompts, len(data))))

    # Process each prompt
    results = []
    for idx, row in enumerate(tqdm(data, desc="Processing DIBT")):
        prompt = row.get("prompt", row.get("instruction", ""))
        if not prompt:
            continue

        # DIBT stores responses as a list of ranked responses or individual columns
        # Try different possible structures
        responses = []
        quality_scores = {dim: [] for dim in DIBT_DIMENSIONS}

        if "responses" in row:
            responses = row["responses"]
        elif "response" in row:
            responses = [row["response"]]

        # Try to get scores for each dimension
        for dim in DIBT_DIMENSIONS:
            if dim in row:
                val = row[dim]
                if isinstance(val, list):
                    quality_scores[dim] = val
                elif isinstance(val, (int, float)):
                    quality_scores[dim] = [val]

        # Ensure responses and scores align
        if not responses:
            continue
        if not any(quality_scores.values()):
            continue

        for i, response in enumerate(responses):
            sig = compute_token_loss_top20(model, tokenizer, prompt, response, device)
            record = {
                "prompt_idx": idx,
                "response_idx": i,
                "prompt": prompt[:200],
                "response": response[:200],
                **sig,
            }
            for dim in DIBT_DIMENSIONS:
                scores_list = quality_scores.get(dim, [])
                record[f"dibt_{dim}"] = float(scores_list[i]) if i < len(scores_list) else None
            results.append(record)

    if not results:
        logger.error("No results extracted. DIBT format may be unexpected. Inspect the first row:")
        logger.error(str(next(iter(data))))
        return

    logger.info(f"Extracted {len(results)} prompt-response pairs")

    # Compute correlations
    print("\n=== DIBT Dimension Correlations ===")
    print(f"{'Dimension':<20} {'n_valid':>8} {'rho(token_top20)':>18} {'rho(IFD)':>18}")
    print("-" * 70)

    corr_table = {}
    signal_keys = ["token_loss_top20", "ifd"]
    for dim in DIBT_DIMENSIONS:
        dim_key = f"dibt_{dim}"
        valid = [r for r in results if r.get(dim_key) is not None and not np.isnan(r.get(dim_key, np.nan))]
        if len(valid) < 10:
            continue
        t20_vals = np.array([r["token_loss_top20"] for r in valid])
        ifd_vals = np.array([r["ifd"] for r in valid])
        dim_vals = np.array([r[dim_key] for r in valid])

        rho_t20, p_t20 = spearmanr(t20_vals, dim_vals)
        rho_ifd, p_ifd = spearmanr(ifd_vals, dim_vals)

        corr_table[dim] = {
            "n": len(valid),
            "rho_token_top20": float(rho_t20),
            "p_token_top20": float(p_t20),
            "rho_ifd": float(rho_ifd),
            "p_ifd": float(p_ifd),
        }
        print(f"{dim:<20} {len(valid):>8} {rho_t20:>+10.4f} (p={p_t20:.3f}) {rho_ifd:>+10.4f} (p={p_ifd:.3f})")

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
