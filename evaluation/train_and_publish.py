"""
Multi-task SFT training — v4
Datasets: GSM8K + Orca-Math + MetaMathQA, argilla/ifeval-like-data (filtered), OpenCodeInstruct
Key change: Tulu-3 replaced entirely with argilla ifeval-like-data (56k verified constraint examples)
No system prompt. Optimizer improvements from v3 retained (weight_decay, grad_clip, warmup).

Usage:
    python evaluation/train_and_publish.py
    python evaluation/train_and_publish.py --num_steps 2000 --checkpoint_name multitask-8b-v4
    python evaluation/train_and_publish.py --no_publish
"""

import argparse
import json
import os
import random

import numpy as np
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "meta-llama/Llama-3.1-8B"
# MODEL = "meta-llama/Llama-3.2-3B"   # Debugging only

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# Data mixture sizes
N_GSM8K  = 7473    # All of GSM8K train
N_ORCA   = 20000   # Orca-Math-200k
N_META   = 20000   # MetaMathQA
N_ARGILLA = None   # Use all 56k verified IFEval-like examples
N_CODE   = 10000   # OpenCodeInstruct

CODE_QUALITY_THRESHOLD = 0.9
SEED = 42

COT_PREFIX = "Let's think step by step.\n\n"


def load_gsm8k():
    """Load all GSM8K train examples with CoT prefix."""
    print("  Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    convos = []
    for ex in ds:
        convos.append([
            {"role": "user",      "content": ex["question"]},
            {"role": "assistant", "content": COT_PREFIX + ex["answer"]},
        ])
    print(f"  GSM8K: {len(convos)} examples")
    return convos


def load_orca_math(n):
    """Sample n examples from Orca-Math-200k."""
    print(f"  Loading Orca-Math (sampling {n})...")
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train", streaming=True)
    convos = []
    for ex in ds:
        convos.append([
            {"role": "user",      "content": ex["question"]},
            {"role": "assistant", "content": COT_PREFIX + ex["answer"]},
        ])
        if len(convos) >= n:
            break
    print(f"  Orca-Math: {len(convos)} examples")
    return convos


def load_metamath(n):
    """Sample n examples from MetaMathQA."""
    print(f"  Loading MetaMathQA (sampling {n})...")
    ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    convos = []
    for ex in ds:
        convos.append([
            {"role": "user",      "content": ex["query"]},
            {"role": "assistant", "content": COT_PREFIX + ex["response"]},
        ])
        if len(convos) >= n:
            break
    print(f"  MetaMathQA: {len(convos)} examples")
    return convos


def load_argilla_ifeval():
    """Load all verified examples from argilla/ifeval-like-data filtered subset."""
    print("  Loading argilla ifeval-like-data (filtered, all examples)...")
    ds = load_dataset("argilla/ifeval-like-data", "filtered", split="train")
    convos = []
    for ex in ds:
        # Only use examples verified at prompt level strict
        if not ex.get("prompt_level_strict_acc", False):
            continue
        convos.append([
            {"role": "user",      "content": ex["prompt"]},
            {"role": "assistant", "content": ex["response"]},
        ])
    print(f"  Argilla IFEval-like: {len(convos)} examples")
    return convos


def load_opencode(n, quality_threshold=CODE_QUALITY_THRESHOLD):
    """Sample n high-quality examples from OpenCodeInstruct."""
    print(f"  Loading OpenCodeInstruct (sampling {n}, quality>={quality_threshold})...")
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
    convos = []
    for ex in ds:
        try:
            score = float(ex.get("average_test_score", 0))
        except (ValueError, TypeError):
            score = 0.0
        if score < quality_threshold:
            continue
        convos.append([
            {"role": "user",      "content": ex["input"]},
            {"role": "assistant", "content": ex["output"]},
        ])
        if len(convos) >= n:
            break
    print(f"  OpenCodeInstruct: {len(convos)} examples")
    return convos


def weighted_sample(datasets_with_weights, seed):
    """
    Sample from multiple datasets according to weights.
    datasets_with_weights: list of (convos, weight) tuples
    """
    random.seed(seed)
    total_weight = sum(w for _, w in datasets_with_weights)
    total_examples = sum(len(c) for c, _ in datasets_with_weights)
    result = []
    for convos, weight in datasets_with_weights:
        n = int(total_examples * weight / total_weight)
        result.extend(random.sample(convos, min(n, len(convos))))
    random.shuffle(result)
    return result


def get_lr(step, base_lr, warmup_steps):
    """Linear warmup then constant learning rate."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def main():
    parser = argparse.ArgumentParser(description="Train, save, and publish a checkpoint")
    parser.add_argument("--num_steps",       type=int,   default=2000,              help="Number of training steps")
    parser.add_argument("--batch_size",      type=int,   default=4,                 help="Batch size")
    parser.add_argument("--lr",              type=float, default=5e-5,              help="Peak learning rate")
    parser.add_argument("--warmup_steps",    type=int,   default=150,               help="LR warmup steps")
    parser.add_argument("--weight_decay",    type=float, default=0.01,              help="Adam weight decay")
    parser.add_argument("--grad_clip_norm",  type=float, default=1.0,               help="Gradient clip norm")
    parser.add_argument("--rank",            type=int,   default=128,               help="LoRA rank")
    parser.add_argument("--max_length",      type=int,   default=1024,              help="Max token length")
    parser.add_argument("--checkpoint_name", type=str,   default="multitask-8b-v4", help="Checkpoint name")
    parser.add_argument("--no_publish",      action="store_true",                   help="Skip publishing")
    args = parser.parse_args()

    # Setup
    print(f"Model: {MODEL}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Load datasets
    print("\nLoading datasets...")
    gsm8k_convos   = load_gsm8k()
    orca_convos    = load_orca_math(N_ORCA)
    meta_convos    = load_metamath(N_META)
    argilla_convos = load_argilla_ifeval()
    code_convos    = load_opencode(N_CODE)

    math_convos = gsm8k_convos + orca_convos + meta_convos
    print(f"\nTotal math examples (GSM8K + Orca + MetaMath): {len(math_convos)}")

    # Weighted sampling
    # argilla gets highest weight — targeted verified IFEval constraint examples
    # math gets moderate weight — already strong but more helps competition ranking
    # code gets lowest weight — already well above target
    datasets_with_weights = [
        (math_convos,    2.0),
        (argilla_convos, 4.0),
        (code_convos,    1.0),
    ]
    all_convos = weighted_sample(datasets_with_weights, SEED)
    print(f"Total conversations after weighted sampling: {len(all_convos)}")

    # Tokenize
    print("Tokenizing...")
    all_data = []
    skipped = 0
    for convo in all_convos:
        try:
            datum = conversation_to_datum(
                convo,
                renderer,
                max_length=args.max_length,
                train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            all_data.append(datum)
        except Exception:
            skipped += 1
    print(f"  {len(all_data)} examples ready ({skipped} skipped)")

    # Create training client
    print(f"\nCreating LoRA training client (rank={args.rank})...")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(
        base_model=MODEL,
        rank=args.rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    print("  Training client ready")

    # Train with LR warmup
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, "
          f"peak_lr={args.lr}, warmup={args.warmup_steps}, "
          f"weight_decay={args.weight_decay}, grad_clip={args.grad_clip_norm})...")

    losses = []
    for step in range(args.num_steps):
        current_lr = get_lr(step, args.lr, args.warmup_steps)

        adam_params = types.AdamParams(
            learning_rate=current_lr,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
        )

        start = (step * args.batch_size) % len(all_data)
        batch = [all_data[i % len(all_data)] for i in range(start, start + args.batch_size)]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future   = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights  = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss     = -np.dot(logprobs, weights) / max(weights.sum(), 1)
        losses.append(loss)

        if (step + 1) % 100 == 0 or step == 0:
            avg_loss = np.mean(losses[-100:])
            print(f"  Step {step+1}/{args.num_steps} | Loss: {loss:.4f} | "
                  f"Avg(100): {avg_loss:.4f} | LR: {current_lr:.2e}")

    # Save checkpoint
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

    # Publish
    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print("  Published successfully!")
    else:
        print("\nSkipping publish (--no_publish).")

    # Save checkpoint info
    info = {
        "checkpoint_path": checkpoint_path,
        "base_model": MODEL,
        "renderer_name": renderer_name,
        "training": {
            "num_steps":         args.num_steps,
            "batch_size":        args.batch_size,
            "learning_rate":     args.lr,
            "warmup_steps":      args.warmup_steps,
            "weight_decay":      args.weight_decay,
            "grad_clip_norm":    args.grad_clip_norm,
            "lora_rank":         args.rank,
            "max_length":        args.max_length,
            "n_gsm8k":           len(gsm8k_convos),
            "n_orca":            len(orca_convos),
            "n_meta":            len(meta_convos),
            "n_argilla":         len(argilla_convos),
            "n_code":            len(code_convos),
            "code_quality":      CODE_QUALITY_THRESHOLD,
            "cot_prefix":        True,
            "system_prompt":     False,
            "weighted_sampling": True,
            "argilla_weight":    4.0,
            "math_weight":       2.0,
            "code_weight":       1.0,
        },
        "published": not args.no_publish,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nNext: evaluate your checkpoint with")
    print(f'  python -m evaluation.eval_all --checkpoint_path "{checkpoint_path}" --base_model {MODEL}')


if __name__ == "__main__":
    main()
