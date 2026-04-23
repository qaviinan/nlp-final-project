"""
Multi-task SFT training — v5b continuation
Resumes from v5 step 8000 checkpoint and trains for another 8000 steps.
Checkpoints saved every 1000 steps.

Usage:
    python evaluation/train_and_publish.py
    python evaluation/train_and_publish.py --num_steps 8000 --checkpoint_name multitask-8b-v5b
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
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# Resume from v5 step 8000 training state
RESUME_FROM = "tinker://5194866b-f63f-5c06-9398-c0f780df94fb:train:0/weights/multitask-8b-v5-step8000"

# Dataset sizes — same as v5
N_GSM8K   = 7473
N_ORCA    = 50000
N_META    = 50000
N_TULU    = 100000
N_ARGILLA = None    # All 56k
N_CODE    = 100000

CODE_QUALITY_THRESHOLD = 0.9
SEED = 42
COT_PREFIX = "Let's think step by step.\n\n"
CHECKPOINT_INTERVAL = 1000


def load_gsm8k():
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


def load_tulu(n):
    print(f"  Loading Tulu-3 (sampling {n})...")
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    convos = []
    for ex in ds:
        convos.append(ex["messages"])
        if len(convos) >= n:
            break
    print(f"  Tulu-3: {len(convos)} examples")
    return convos


def load_argilla_ifeval():
    print("  Loading Argilla IFEval-like (all verified)...")
    ds = load_dataset("argilla/ifeval-like-data", "filtered", split="train")
    convos = []
    for ex in ds:
        if not ex.get("prompt_level_strict_acc", False):
            continue
        convos.append([
            {"role": "user",      "content": ex["prompt"]},
            {"role": "assistant", "content": ex["response"]},
        ])
    print(f"  Argilla IFEval-like: {len(convos)} examples")
    return convos


def load_opencode(n, quality_threshold=CODE_QUALITY_THRESHOLD):
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
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def save_checkpoint(tc, sc, name, publish=True):
    print(f"\n  Saving training state '{name}'...")
    state = tc.save_state(name=name).result()
    training_state_path = state.path
    print(f"  Training state: {training_state_path}")

    print(f"  Saving sampler checkpoint '{name}'...")
    ckpt = tc.save_weights_for_sampler(name=name).result()
    checkpoint_path = ckpt.path
    print(f"  Sampler checkpoint: {checkpoint_path}")

    if publish:
        print(f"  Publishing '{name}'...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
        print(f"  Published!")

    return checkpoint_path, training_state_path


def main():
    parser = argparse.ArgumentParser(description="Continue training from v5 step 8000")
    parser.add_argument("--num_steps",           type=int,   default=8000)
    parser.add_argument("--batch_size",           type=int,   default=4)
    parser.add_argument("--lr",                   type=float, default=3e-5,
                        help="Slightly lower LR for continuation run")
    parser.add_argument("--warmup_steps",         type=int,   default=100)
    parser.add_argument("--weight_decay",         type=float, default=0.01)
    parser.add_argument("--grad_clip_norm",       type=float, default=1.0)
    parser.add_argument("--rank",                 type=int,   default=128)
    parser.add_argument("--max_length",           type=int,   default=1024)
    parser.add_argument("--checkpoint_name",      type=str,   default="multitask-8b-v5b")
    parser.add_argument("--checkpoint_interval",  type=int,   default=CHECKPOINT_INTERVAL)
    parser.add_argument("--resume_from",          type=str,   default=RESUME_FROM)
    parser.add_argument("--no_publish",           action="store_true")
    args = parser.parse_args()

    # Setup
    print(f"Model: {MODEL}")
    print(f"Resuming from: {args.resume_from}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Load datasets
    print("\nLoading datasets...")
    gsm8k_convos   = load_gsm8k()
    orca_convos    = load_orca_math(N_ORCA)
    meta_convos    = load_metamath(N_META)
    tulu_convos    = load_tulu(N_TULU)
    argilla_convos = load_argilla_ifeval()
    code_convos    = load_opencode(N_CODE)

    math_convos = gsm8k_convos + orca_convos + meta_convos
    if_convos   = tulu_convos + argilla_convos
    print(f"\nTotal math:  {len(math_convos)}")
    print(f"Total IF:    {len(if_convos)}")
    print(f"Total code:  {len(code_convos)}")
    print(f"Grand total: {len(math_convos) + len(if_convos) + len(code_convos)}")

    datasets_with_weights = [
        (math_convos, 2.0),
        (if_convos,   4.0),
        (code_convos, 2.0),
    ]
    # Different seed so we see different examples than v5
    all_convos = weighted_sample(datasets_with_weights, seed=SEED + 1)
    print(f"\nTotal after weighted sampling: {len(all_convos)}")

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

    # Resume from checkpoint
    print(f"\nResuming training from checkpoint...")
    sc = tinker.ServiceClient()
    tc = sc.create_training_client_from_state(args.resume_from)
    print("  Training client loaded from checkpoint")

    # Train
    print(f"\nContinuing for {args.num_steps} steps "
          f"(batch={args.batch_size}, lr={args.lr}, warmup={args.warmup_steps}, "
          f"checkpoint every {args.checkpoint_interval} steps)...")

    losses = []
    checkpoints = []

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

        if (step + 1) % args.checkpoint_interval == 0:
            global_step = 8000 + step + 1
            ckpt_name = f"{args.checkpoint_name}-step{global_step}"
            checkpoint_path, training_state_path = save_checkpoint(
                tc, sc, ckpt_name, publish=not args.no_publish
            )
            checkpoints.append({
                "step": global_step,
                "checkpoint_path": checkpoint_path,
                "training_state_path": training_state_path,
                "avg_loss": float(np.mean(losses[-100:])),
            })
            print(f"  Checkpoint saved at global step {global_step}\n")

    # Final checkpoint
    print("\nSaving final checkpoint...")
    checkpoint_path, training_state_path = save_checkpoint(
        tc, sc, args.checkpoint_name, publish=not args.no_publish
    )
    checkpoints.append({
        "step": 8000 + args.num_steps,
        "checkpoint_path": checkpoint_path,
        "training_state_path": training_state_path,
        "avg_loss": float(np.mean(losses[-100:])),
    })

    # Save checkpoint info
    info = {
        "final_checkpoint_path":       checkpoint_path,
        "final_training_state_path":   training_state_path,
        "base_model":                  MODEL,
        "renderer_name":               renderer_name,
        "resumed_from":                args.resume_from,
        "global_steps":                8000 + args.num_steps,
        "checkpoints":                 checkpoints,
        "training": {
            "num_steps":               args.num_steps,
            "batch_size":              args.batch_size,
            "learning_rate":           args.lr,
            "warmup_steps":            args.warmup_steps,
            "weight_decay":            args.weight_decay,
            "grad_clip_norm":          args.grad_clip_norm,
            "lora_rank":               args.rank,
            "max_length":              args.max_length,
            "checkpoint_interval":     args.checkpoint_interval,
            "n_gsm8k":                 len(gsm8k_convos),
            "n_orca":                  len(orca_convos),
            "n_meta":                  len(meta_convos),
            "n_tulu":                  len(tulu_convos),
            "n_argilla":               len(argilla_convos),
            "n_code":                  len(code_convos),
            "code_quality":            CODE_QUALITY_THRESHOLD,
            "seed":                    SEED + 1,
        },
        "published": not args.no_publish,
    }
    info_path = os.path.join(EVAL_DIR, "checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nAll checkpoint info saved to {info_path}")

    print("\nEvaluate all checkpoints with:")
    for ckpt in checkpoints:
        print(f"  Step {ckpt['step']:6d}: python -m evaluation.eval_all "
              f"--checkpoint_path \"{ckpt['checkpoint_path']}\" "
              f"--base_model {MODEL}")


if __name__ == "__main__":
    main()