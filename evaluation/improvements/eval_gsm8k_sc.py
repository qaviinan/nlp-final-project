"""
GSM8K self-consistency evaluation (Wang et al., 2022).

Samples N CoTs per problem at temperature T, extracts the final numeric answer
from each, and majority-votes across samples. Reliably adds +5-15 pp over greedy
on GSM8K for 7-13B models.

Usage:
    python -m evaluation.improvements.eval_gsm8k_sc \
        --checkpoint_path "tinker://..." \
        --base_model meta-llama/Llama-3.1-8B \
        --n_samples 16 --temperature 0.6

    # Quick smoke test:
    python -m evaluation.improvements.eval_gsm8k_sc \
        --base_model meta-llama/Llama-3.2-3B --n_samples 4 --limit 10
"""

import argparse
import asyncio
import json
import logging
import os
import re
from collections import Counter
from typing import Optional

import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# GSM8K canonical answer marker is '####'; we also accept \boxed{} and a final number.
ANSWER_RE = re.compile(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)")
BOXED_RE = re.compile(r"\\boxed\{\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*\}")
NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def normalize_number(s: str) -> Optional[str]:
    """Strip commas, trailing zeros; return canonical numeric string or None."""
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        # Normalize trailing zeros / avoid float-formatting drift.
        return ("%g" % f)
    except ValueError:
        return None


def extract_answer(text: str) -> Optional[str]:
    """Preference: '####' marker -> '\\boxed{}' -> last number in text."""
    m = ANSWER_RE.search(text)
    if m:
        return normalize_number(m.group(1))
    m = BOXED_RE.search(text)
    if m:
        return normalize_number(m.group(1))
    nums = NUMBER_RE.findall(text)
    if nums:
        return normalize_number(nums[-1])
    return None


def extract_gt_answer(answer_field: str) -> Optional[str]:
    """GSM8K train/test answer_field ends with '#### N'."""
    m = ANSWER_RE.search(answer_field)
    if m:
        return normalize_number(m.group(1))
    return None


def majority_vote(answers):
    """Return (top_answer, vote_count, total_non_null)."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0, 0
    counter = Counter(valid)
    top, cnt = counter.most_common(1)[0]
    return top, cnt, len(valid)


async def sample_one_problem(
    sampling_client,
    sampling_params,
    renderer,
    question: str,
    n_samples: int,
    system_prompt: Optional[str],
):
    convo = []
    if system_prompt:
        convo.append({"role": "system", "content": system_prompt})
    convo.append({"role": "user", "content": question})
    prompt_tokens = renderer.build_generation_prompt(convo)
    fut = sampling_client.sample(
        num_samples=n_samples,
        prompt=prompt_tokens,
        sampling_params=sampling_params,
    )
    result = fut.result()
    return result.sequences


async def run(args):
    tokenizer = get_tokenizer(args.base_model)
    renderer_name = args.renderer_name or model_info.get_recommended_renderer_name(args.base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Model: {args.base_model} | Renderer: {renderer_name}")
    logger.info(f"N samples = {args.n_samples}, T = {args.temperature}, top_p = {args.top_p}")

    sc = tinker.ServiceClient()
    if args.checkpoint_path:
        sampling_client = sc.create_sampling_client(model_path=args.checkpoint_path)
    else:
        sampling_client = sc.create_sampling_client(base_model=args.base_model)

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    logger.info(f"Loaded {len(ds)} test problems")

    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=args.stop or None,
    )

    system_prompt = args.system_prompt
    if args.default_system_prompt and not system_prompt:
        system_prompt = (
            "Solve the math word problem step by step. "
            "After your reasoning, write the final numeric answer on its own line "
            "in the exact form '#### N' where N is the number only."
        )

    records = []
    correct_sc = 0
    correct_greedy = 0  # first sample only (proxy for greedy behavior)

    sem = asyncio.Semaphore(args.concurrency)

    async def process(i, ex):
        async with sem:
            try:
                seqs = await sample_one_problem(
                    sampling_client, sampling_params, renderer,
                    ex["question"], args.n_samples, system_prompt,
                )
            except Exception as e:
                logger.warning(f"Problem {i} sampling failed: {e}")
                return None

            texts = [tokenizer.decode(s.tokens) for s in seqs]
            preds = [extract_answer(t) for t in texts]
            gt = extract_gt_answer(ex["answer"])
            sc_ans, sc_votes, sc_valid = majority_vote(preds)

            rec = {
                "id": i,
                "question": ex["question"],
                "gt": gt,
                "preds": preds,
                "sc_answer": sc_ans,
                "sc_votes": sc_votes,
                "sc_valid": sc_valid,
                "first_pred": preds[0] if preds else None,
                "sc_correct": int(sc_ans is not None and sc_ans == gt),
                "first_correct": int(preds and preds[0] is not None and preds[0] == gt),
            }
            if args.save_texts:
                rec["texts"] = texts
            return rec

    tasks = [process(i, ex) for i, ex in enumerate(ds)]
    for fut in asyncio.as_completed(tasks):
        rec = await fut
        if rec is None:
            continue
        records.append(rec)
        correct_sc += rec["sc_correct"]
        correct_greedy += rec["first_correct"]
        if len(records) % 25 == 0 or len(records) == len(ds):
            logger.info(
                f"  Progress {len(records)}/{len(ds)} | "
                f"SC acc = {correct_sc / len(records):.3f} | "
                f"First-sample acc = {correct_greedy / len(records):.3f}"
            )

    records.sort(key=lambda r: r["id"])
    total = len(records)
    sc_acc = correct_sc / total if total else 0.0
    greedy_acc = correct_greedy / total if total else 0.0

    out = {
        "task": "gsm8k_self_consistency",
        "base_model": args.base_model,
        "checkpoint_path": args.checkpoint_path,
        "renderer_name": renderer_name,
        "settings": {
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "system_prompt": system_prompt,
        },
        "metrics": {
            "self_consistency_accuracy": sc_acc,
            "first_sample_accuracy": greedy_acc,
            "n_problems": total,
        },
        "samples": records,
    }

    out_path = args.output_path or os.path.join(
        EVAL_DIR, "improvements_results", "gsm8k_sc.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print("GSM8K SELF-CONSISTENCY")
    print("=" * 60)
    print(f"  Problems evaluated : {total}")
    print(f"  N per problem      : {args.n_samples}")
    print(f"  Temperature        : {args.temperature}")
    print(f"  SC accuracy        : {sc_acc:.4f}")
    print(f"  First-sample acc   : {greedy_acc:.4f}")
    print(f"  Delta (SC - first) : {sc_acc - greedy_acc:+.4f}")
    print(f"  Saved to           : {out_path}")
    print("=" * 60)
    return out


def main():
    p = argparse.ArgumentParser(description="GSM8K self-consistency evaluation")
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--renderer_name", type=str, default=None)
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--stop", nargs="*", default=None,
                   help="Optional stop strings (e.g. 'User:' to prevent chat continuations)")
    p.add_argument("--system_prompt", type=str, default=None)
    p.add_argument("--default_system_prompt", action="store_true",
                   help="Use built-in system prompt requesting '#### N' format")
    p.add_argument("--save_texts", action="store_true",
                   help="Persist full sample texts (large output file)")
    p.add_argument("--output_path", type=str, default=None)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
