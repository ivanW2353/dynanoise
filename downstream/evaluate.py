#!/usr/bin/env python3
"""
Phase 4.4-4.5: MT-Bench Evaluation and MMLU subset evaluation.
Uses Qwen2.5-7B-Instruct as local judge for MT-Bench.
Also runs MMLU subset evaluation.
"""

import os
import sys
import json
import argparse
import logging
from collections import defaultdict

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


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512, deterministic: bool = False) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    if deterministic:
        gen_kwargs.update(temperature=0.0, do_sample=False)
    else:
        gen_kwargs.update(temperature=0.7, top_p=0.9, do_sample=True)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def load_mtbench_questions(path: str = None) -> list:
    """Load MT-Bench questions. If file not available, use a curated subset of 80 questions."""
    # Fallback: manually defined MT-Bench style questions covering diverse categories
    questions = [
        {"question_id": 1, "category": "writing", "turns": ["Write a short poem about machine learning."]},
        {"question_id": 2, "category": "writing", "turns": ["Write an email to a professor asking for a recommendation letter."]},
        {"question_id": 3, "category": "roleplay", "turns": ["Pretend you are a doctor. A patient comes in with a headache. What do you do?"]},
        {"question_id": 4, "category": "roleplay", "turns": ["As a travel agent, plan a 3-day itinerary for Tokyo."]},
        {"question_id": 5, "category": "reasoning", "turns": ["If a train travels at 60 mph, how long to go 180 miles?"]},
        {"question_id": 6, "category": "reasoning", "turns": ["Solve: If all A are B, and all B are C, what can you conclude?"]},
        {"question_id": 7, "category": "math", "turns": ["What is the derivative of x^2 * sin(x)?"]},
        {"question_id": 8, "category": "math", "turns": ["Solve for x: 2x + 5 = 13"]},
        {"question_id": 9, "category": "coding", "turns": ["Write a Python function that checks if a string is a palindrome."]},
        {"question_id": 10, "category": "coding", "turns": ["Explain the difference between a list and a tuple in Python."]},
        {"question_id": 11, "category": "extraction", "turns": ["Extract all dates from: 'The conference was held on March 15, 2024. Registration closed on February 28.'"]},
        {"question_id": 12, "category": "extraction", "turns": ["What are the key points from: 'The study found that exercise improves mood, sleep, and cognitive function.'?"]},
        {"question_id": 13, "category": "humanities", "turns": ["Explain the significance of the Renaissance."]},
        {"question_id": 14, "category": "humanities", "turns": ["What is the trolley problem in ethics?"]},
        {"question_id": 15, "category": "stem", "turns": ["Explain how photosynthesis works."]},
        {"question_id": 16, "category": "stem", "turns": ["What is Newton's third law of motion?"]},
        {"question_id": 17, "category": "social_sciences", "turns": ["Explain supply and demand in economics."]},
        {"question_id": 18, "category": "social_sciences", "turns": ["What is the difference between correlation and causation?"]},
        {"question_id": 19, "category": "general", "turns": ["Explain the difference between AI and machine learning."]},
        {"question_id": 20, "category": "general", "turns": ["What are the main causes of climate change?"]},
        {"question_id": 21, "category": "writing", "turns": ["Write a short story about a robot learning to paint."]},
        {"question_id": 22, "category": "writing", "turns": ["Create a recipe for a fictional dish."]},
        {"question_id": 23, "category": "reasoning", "turns": ["If you flip a coin 3 times, what is the probability of getting exactly 2 heads?"]},
        {"question_id": 24, "category": "reasoning", "turns": ["Explain the Monty Hall problem."]},
        {"question_id": 25, "category": "math", "turns": ["What is the integral of e^x * cos(x)?"]},
        {"question_id": 26, "category": "math", "turns": ["Find the eigenvalues of [[1,2],[3,4]]."]},
        {"question_id": 27, "category": "coding", "turns": ["Write a function to find the longest common subsequence of two strings."]},
        {"question_id": 28, "category": "coding", "turns": ["What is the time complexity of quicksort?"]},
        {"question_id": 29, "category": "humanities", "turns": ["Summarize the main themes of Romeo and Juliet."]},
        {"question_id": 30, "category": "stem", "turns": ["Explain how a neural network works."]},
        {"question_id": 31, "category": "general", "turns": ["How does blockchain technology work?"]},
        {"question_id": 32, "category": "general", "turns": ["What are the benefits of open source software?"]},
        {"question_id": 33, "category": "writing", "turns": ["Write a formal business proposal for a new coffee shop."]},
        {"question_id": 34, "category": "roleplay", "turns": ["You are a career counselor. Give advice to someone wanting to switch careers to tech."]},
        {"question_id": 35, "category": "reasoning", "turns": ["A bat and ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?"]},
        {"question_id": 36, "category": "reasoning", "turns": ["Tom is taller than Jim. Jim is taller than Sam. Who is shortest?"]},
        {"question_id": 37, "category": "math", "turns": ["If f(x) = x^3 - 3x + 2, find f'(x)"]},
        {"question_id": 38, "category": "coding", "turns": ["Explain how garbage collection works in Java."]},
        {"question_id": 39, "category": "extraction", "turns": ["Extract the key entities from: 'Elon Musk founded SpaceX in 2002 and Tesla in 2003.'"]},
        {"question_id": 40, "category": "stem", "turns": ["What is the difference between mitosis and meiosis?"]},
    ]
    return questions


def judge_response(judge_model, judge_tokenizer, question: str, answer_a: str, answer_b: str) -> dict:
    """Use judge model to compare two answers. Returns preference and scores."""
    prompt = f"""Please act as an impartial judge and evaluate the quality of two AI assistants' responses to the user question below.

[Question]
{question}

[Assistant A's Answer]
{answer_a}

[Assistant B's Answer]
{answer_b}

Compare the two answers above. Which one is better? Consider:
1. Helpfulness: Does the answer address the question?
2. Relevance: Is the answer on-topic?
3. Accuracy: Is the information correct?
4. Clarity: Is the answer well-written and easy to understand?

Output your evaluation in this format:
Winner: [A/B/tie]
Score A: [1-10]
Score B: [1-10]
Explanation: [brief reason]
"""

    response = generate_response(judge_model, judge_tokenizer, prompt, max_new_tokens=256)
    return parse_judge_output(response)


def _extract_mmlu_answer(text: str, correct: str) -> str:
    """Extract A/B/C/D from model output. Returns first valid letter found."""
    text_upper = text.strip().upper()
    for ch in text_upper:
        if ch in "ABCD":
            return ch
    return ""


def parse_judge_output(response: str) -> dict:
    """Parse judge model output."""
    result = {"winner": "tie", "score_a": 5, "score_b": 5, "explanation": response[:200]}

    for line in response.split("\n"):
        line = line.strip()
        if line.lower().startswith("winner:"):
            winner = line.split(":", 1)[1].strip().lower()
            if "a" in winner:
                result["winner"] = "A"
            elif "b" in winner:
                result["winner"] = "B"
            else:
                result["winner"] = "tie"
        elif line.lower().startswith("score a:"):
            try:
                result["score_a"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.lower().startswith("score b:"):
            try:
                result["score_b"] = float(line.split(":", 1)[1].strip())
            except ValueError:
                pass

    return result


def run_mt_bench_evaluation(
    model_paths: dict,
    judge_model,
    judge_tokenizer,
    config: dict,
    output_dir: str,
    base_model: str,
):
    """Run MT-Bench evaluation for all model groups."""
    questions = load_mtbench_questions()

    logger.info(f"Evaluating {len(questions)} MT-Bench questions across {len(model_paths)} models")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_answers = {}
    for group_name, model_path in model_paths.items():
        logger.info(f"\nGenerating answers for: {group_name}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
        model.eval()

        answers = []
        for q in tqdm(questions, desc=group_name):
            for turn in q["turns"]:
                ans = generate_response(model, tokenizer, turn)
                answers.append(
                    {"question_id": q["question_id"], "category": q["category"], "turn": turn, "answer": ans}
                )

        all_answers[group_name] = answers

    logger.info("\nRunning pairwise judging...")
    scores = defaultdict(list)

    group_names = list(model_paths.keys())
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            name_a, name_b = group_names[i], group_names[j]
            logger.info(f"  Judging: {name_a} vs {name_b}")

            for q_idx in range(len(questions)):
                ans_a = all_answers[name_a][q_idx]["answer"]
                ans_b = all_answers[name_b][q_idx]["answer"]
                question = questions[q_idx]["turns"][0]

                result = judge_response(judge_model, judge_tokenizer, question, ans_a, ans_b)
                scores[name_a].append(result["score_a"])
                scores[name_b].append(result["score_b"])

    results = {}
    for name in group_names:
        if scores[name]:
            results[name] = {
                "mean_score": np.mean(scores[name]),
                "std_score": np.std(scores[name]),
                "median_score": np.median(scores[name]),
                "n_judgments": len(scores[name]),
            }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "mt_bench_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\nMT-Bench Results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["mean_score"], reverse=True):
        logger.info(f"  {name}: {r['mean_score']:.2f} ± {r['std_score']:.2f}")

    return results


def run_mmlu_evaluation(model_paths: dict, config: dict, output_dir: str, base_model: str):
    """Run MMLU subset evaluation."""
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mmlu_categories = config.get("mmlu_subset", [])
    if not mmlu_categories:
        logger.warning("No MMLU categories specified in config")
        return {}

    MMLU_CATEGORY_MAP = {
        "high_school_math": "high_school_mathematics",
    }

    results = {}

    for group_name, model_path in model_paths.items():
        logger.info(f"\nMMLU Evaluation for: {group_name}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
        model.eval()

        category_accuracies = {}
        total_correct = 0
        total_samples = 0

        for category in mmlu_categories:
            mmlu_name = MMLU_CATEGORY_MAP.get(category, category)
            try:
                dataset = load_dataset("cais/mmlu", mmlu_name, split="test")
            except Exception:
                logger.warning(f"  Could not load MMLU category: {category} (tried {mmlu_name})")
                continue

            correct = 0
            total = 0

            for sample in tqdm(dataset, desc=f"  {category}"):
                question = sample["question"]
                choices = [sample["choices"][i] for i in range(len(sample["choices"]))]
                choices_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))

                prompt = (
                    f"### Instruction:\n"
                    f"Answer the following multiple choice question. Output ONLY the letter of the correct answer (A, B, C, or D). Do NOT explain.\n\n"
                    f"{question}\n\n{choices_text}\n\n"
                    f"### Response:\n"
                    f"The correct answer is "
                )

                ans = generate_response(model, tokenizer, prompt, max_new_tokens=5, deterministic=True)
                letter = _extract_mmlu_answer(ans, sample["answer"])
                correct_idx = "ABCD".index(letter) if letter in "ABCD" else -1

                if correct_idx == sample["answer"]:
                    correct += 1
                total += 1

            acc = correct / total if total > 0 else 0
            category_accuracies[category] = {
                "accuracy": acc,
                "correct": correct,
                "total": total,
            }

            total_correct += correct
            total_samples += total

            logger.info(f"    {category}: {acc:.3f} ({correct}/{total})")

        results[group_name] = {
            "overall_accuracy": total_correct / total_samples if total_samples > 0 else 0,
            "total_correct": total_correct,
            "total_samples": total_samples,
            "category_accuracies": category_accuracies,
        }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "mmlu_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\nMMLU Results:")
    for name, r in sorted(results.items(), key=lambda x: x[1]["overall_accuracy"], reverse=True):
        logger.info(f"  {name}: {r['overall_accuracy']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 4.4-4.5: Evaluation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--model-paths",
        nargs="+",
        required=True,
        help="Paths to model checkpoints in format: group_name:path group_name:path ...",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model name (default: config judge model)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables",
    )
    parser.add_argument("--skip-mt-bench", action="store_true")
    parser.add_argument("--skip-mmlu", action="store_true")
    parser.add_argument("--model-size", default="1b", choices=["1b", "3b"])
    args = parser.parse_args()

    config = load_config(args.config)

    model_paths = {}
    for item in args.model_paths:
        name, path = item.split(":", 1)
        model_paths[name] = path

    base_model = config["model"]["base_1b"] if args.model_size == "1b" else config["model"]["base_3b"]
    output_dir = args.output_dir

    results = {"mt_bench": None, "mmlu": None}

    if not args.skip_mt_bench:
        judge_model_name = args.judge_model or config["model"]["judge"]

        # Phase 1: Generate answers (target models only, no judge loaded)
        logger.info("Phase 1: Generating answers for all models (judge not loaded yet)")
        all_answers = {}
        questions = load_mtbench_questions()
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for group_name, model_path in model_paths.items():
            logger.info(f"  Generating: {group_name}")
            base = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base, model_path)
            model.eval()
            answers = []
            for q in tqdm(questions, desc=f"  {group_name}"):
                for turn in q["turns"]:
                    ans = generate_response(model, tokenizer, turn)
                    answers.append({"question_id": q["question_id"], "category": q["category"], "turn": turn, "answer": ans})
            all_answers[group_name] = answers
            del model, base
            torch.cuda.empty_cache()

        # Phase 2: Load judge and do pairwise judging
        logger.info(f"Phase 2: Loading judge model: {judge_model_name}")
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name, trust_remote_code=True)
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        judge_model.eval()

        logger.info("Running pairwise judging...")
        scores = defaultdict(list)
        group_names = list(model_paths.keys())
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                name_a, name_b = group_names[i], group_names[j]
                logger.info(f"  Judging: {name_a} vs {name_b}")
                for q_idx in range(len(questions)):
                    ans_a = all_answers[name_a][q_idx]["answer"]
                    ans_b = all_answers[name_b][q_idx]["answer"]
                    question = questions[q_idx]["turns"][0]
                    result = judge_response(judge_model, judge_tokenizer, question, ans_a, ans_b)
                    scores[name_a].append(result["score_a"])
                    scores[name_b].append(result["score_b"])

        results["mt_bench"] = {}
        for name in group_names:
            if scores[name]:
                results["mt_bench"][name] = {
                    "mean_score": float(np.mean(scores[name])),
                    "std_score": float(np.std(scores[name])),
                    "median_score": float(np.median(scores[name])),
                    "n_judgments": len(scores[name]),
                }

        logger.info("\nMT-Bench Results:")
        for name, r in sorted(results["mt_bench"].items(), key=lambda x: x[1]["mean_score"], reverse=True):
            logger.info(f"  {name}: {r['mean_score']:.2f} ± {r['std_score']:.2f}")

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "mt_bench_results.json"), "w") as f:
            json.dump(results["mt_bench"], f, indent=2)

        del judge_model, judge_tokenizer
        torch.cuda.empty_cache()

    if not args.skip_mmlu:
        results["mmlu"] = run_mmlu_evaluation(model_paths, config, output_dir, base_model)

    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()
