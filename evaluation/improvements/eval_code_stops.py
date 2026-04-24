"""
HumanEval evaluation with code-specific stop sequences and post-processing.

Chat-SFT Llama tends to continue past the target function into test code,
explanations, or extra examples that break the Inspect sandbox's exec. This
evaluator:
  1. Applies stop sequences that fire on common "I'm done" markers.
  2. Post-processes: strips code fences, trims to the first top-level def/class,
     removes trailing prose.
  3. Runs each candidate against the HumanEval reference tests locally
     using subprocess with a timeout.

Usage:
    python -m evaluation.improvements.eval_code_stops \
        --checkpoint_path "tinker://..." \
        --base_model meta-llama/Llama-3.1-8B

    # Quick smoke test on 10 problems:
    python -m evaluation.improvements.eval_code_stops \
        --base_model meta-llama/Llama-3.2-3B --limit 10
"""

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import textwrap
from typing import Optional

import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_STOPS = [
    "\nclass ",
    "\nif __name__",
    "\n# Test",
    "\n# test",
    "\nprint(",
    "\nassert ",
    "\nUser:",
    "\nuser:",
    "</code>",
]

CODE_SYSTEM_PROMPT = (
    "You are a Python coding assistant. Complete the function below. "
    "Return ONLY a single self-contained Python code block with the completed "
    "function. Do not include explanations, tests, or examples."
)


def extract_code(text: str, entry_point: str) -> str:
    """Return the first body we can plausibly execute for `entry_point`."""
    # Prefer fenced ```python blocks.
    fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    body = fence.group(1) if fence else text

    # If we see the entry-point function def, keep from there through the next
    # top-level def/class or the end.
    dp = re.search(rf"^def\s+{re.escape(entry_point)}\s*\(", body, re.MULTILINE)
    if dp:
        start = dp.start()
        rest = body[start:]
        # Find next top-level def/class *after* the first line.
        next_top = re.search(r"^\S", rest[1:], re.MULTILINE)
        if next_top:
            # Keep through end of the function block: first line + indented lines.
            lines = rest.splitlines()
            out_lines = [lines[0]]
            for ln in lines[1:]:
                if ln.strip() == "" or ln.startswith((" ", "\t")):
                    out_lines.append(ln)
                else:
                    break
            body = "\n".join(out_lines)
        else:
            body = rest

    # Strip any leftover triple-backtick or prose.
    body = body.replace("```", "")
    return body.rstrip() + "\n"


def run_humaneval_problem(prompt: str, completion: str, test: str,
                          entry_point: str, timeout: float = 10.0) -> dict:
    """Execute prompt + completion + test in an isolated subprocess."""
    program = (
        prompt
        + "\n"
        + completion
        + "\n"
        + test
        + f"\ncheck({entry_point})\n"
    )
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(program)
            path = f.name
        try:
            proc = subprocess.run(
                ["python", path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            ok = proc.returncode == 0
            return {
                "passed": ok,
                "returncode": proc.returncode,
                "stderr": proc.stderr[-500:] if proc.stderr else "",
            }
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except subprocess.TimeoutExpired:
        return {"passed": False, "returncode": -1, "stderr": "timeout"}
    except Exception as e:
        return {"passed": False, "returncode": -2, "stderr": str(e)}


async def run(args):
    tokenizer = get_tokenizer(args.base_model)
    renderer_name = args.renderer_name or model_info.get_recommended_renderer_name(args.base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Model: {args.base_model} | Renderer: {renderer_name}")

    sc = tinker.ServiceClient()
    if args.checkpoint_path:
        sampling_client = sc.create_sampling_client(model_path=args.checkpoint_path)
    else:
        sampling_client = sc.create_sampling_client(base_model=args.base_model)

    ds = load_dataset("openai/openai_humaneval", split="test")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    logger.info(f"Loaded {len(ds)} HumanEval problems")

    stops = list(DEFAULT_STOPS)
    if args.extra_stops:
        stops.extend(args.extra_stops)

    system_prompt = args.system_prompt
    if args.default_system_prompt and not system_prompt:
        system_prompt = CODE_SYSTEM_PROMPT

    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=stops if args.use_stops else None,
    )

    sem = asyncio.Semaphore(args.concurrency)

    async def process(ex):
        async with sem:
            convo = []
            if system_prompt:
                convo.append({"role": "system", "content": system_prompt})
            convo.append({"role": "user", "content": ex["prompt"]})

            prompt_tokens = renderer.build_generation_prompt(convo)
            try:
                fut = sampling_client.sample(
                    num_samples=1,
                    prompt=prompt_tokens,
                    sampling_params=sampling_params,
                )
                res = fut.result()
                raw = tokenizer.decode(res.sequences[0].tokens)
            except Exception as e:
                return {"task_id": ex["task_id"], "passed": False, "error": str(e), "completion": ""}

            completion = extract_code(raw, ex["entry_point"]) if args.postprocess else raw
            run_res = run_humaneval_problem(
                prompt=ex["prompt"],
                completion=completion,
                test=ex["test"],
                entry_point=ex["entry_point"],
                timeout=args.timeout,
            )
            return {
                "task_id": ex["task_id"],
                "entry_point": ex["entry_point"],
                "raw": raw if args.save_raw else None,
                "completion": completion,
                "passed": run_res["passed"],
                "stderr": run_res.get("stderr", ""),
            }

    tasks = [process(ex) for ex in ds]
    records = []
    passed = 0
    for fut in asyncio.as_completed(tasks):
        rec = await fut
        records.append(rec)
        passed += int(rec.get("passed", False))
        if len(records) % 20 == 0 or len(records) == len(ds):
            logger.info(f"  {len(records)}/{len(ds)} | pass@1 so far = {passed/len(records):.3f}")

    total = len(records)
    acc = passed / total if total else 0.0
    out = {
        "task": "humaneval_stops",
        "base_model": args.base_model,
        "checkpoint_path": args.checkpoint_path,
        "settings": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "use_stops": args.use_stops,
            "stops": stops if args.use_stops else [],
            "postprocess": args.postprocess,
            "system_prompt": system_prompt,
        },
        "metrics": {"pass@1": acc, "n": total, "n_passed": passed},
        "samples": records,
    }
    out_path = args.output_path or os.path.join(
        EVAL_DIR, "improvements_results", "humaneval_stops.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print("HUMANEVAL (stops + post-processing)")
    print("=" * 60)
    print(f"  Problems : {total}")
    print(f"  pass@1   : {acc:.4f}")
    print(f"  Saved to : {out_path}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description="HumanEval with stop sequences and post-processing")
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--renderer_name", type=str, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=768)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--use_stops", action="store_true", default=True)
    p.add_argument("--no_stops", action="store_false", dest="use_stops")
    p.add_argument("--extra_stops", nargs="*", default=None)
    p.add_argument("--postprocess", action="store_true", default=True)
    p.add_argument("--no_postprocess", action="store_false", dest="postprocess")
    p.add_argument("--system_prompt", type=str, default=None)
    p.add_argument("--default_system_prompt", action="store_true")
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--save_raw", action="store_true")
    p.add_argument("--output_path", type=str, default=None)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
