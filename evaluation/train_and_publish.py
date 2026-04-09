"""
Multi-task SFT training on GSM8K, Tulu-3, and OpenCodeInstruct.
Targets: IFEval >= 45%, GSM8K >= 50%, HumanEval >= 30%

Usage:
    python evaluation/train_and_publish.py
    python evaluation/train_and_publish.py --num_steps 500 --rank 64
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

MODEL = "meta-llama/Llama-3.2-3B"
# MODEL = "meta-llama/Llama-3.2-1B"    # Smaller, faster for development
# MODEL = "meta-llama/Llama-3.1-8B"    # Recommended for final submission

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

# Number of examples to sample from each dataset
N_MATH = 7473       # Use all of GSM8K train
N_IF   = 7000       # Sample from Tulu-3
N_CODE = 7000       # Sample from OpenCodeInstruct

SEED = 42


def load_gsm8k():
    """Load GSM8K and convert to conversation format."""
    print("  Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    convos = []
    for ex in ds:
        convos.append([
            {"role": "user",      "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ])
    print(f"  GSM8K: {len(convos)} examples")
    return convos


def load_tulu(n):
    """Sample n examples from Tulu-3 SFT mixture (already in messages format)."""
    print(f"  Loading Tulu-3 (sampling {n})...")
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    convos = []
    for ex in ds:
        convos.append(ex["messages"])
        if len(convos) >= n:
            break
    print(f"  Tulu-3: {len(convos)} examples")
    return convos


def load_opencode(n):
    """Sample n examples from OpenCodeInstruct and convert to conversation format."""
    print(f"  Loading OpenCodeInstruct (sampling {n})...")
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
    convos = []
    for ex in ds:
        # Only use high-quality examples (average test score >= 0.8)
        try:
            score = float(ex.get("average_test_score", 0))
        except (ValueError, TypeError):
            score = 0.0
        if score < 0.8:
            continue
        convos.append([
            {"role": "user",      "content": ex["input"]},
            {"role": "assistant", "content": ex["output"]},
        ])
        if len(convos) >= n:
            break
    print(f"  OpenCodeInstruct: {len(convos)} examples")
    return convos


def main():
    parser = argparse.ArgumentParser(description="Train, save, and publish a checkpoint")
    parser.add_argument("--num_steps",       type=int,   default=500,    help="Number of training steps")
    parser.add_argument("--batch_size",      type=int,   default=4,      help="Batch size")
    parser.add_argument("--lr",              type=float, default=1e-4,   help="Learning rate")
    parser.add_argument("--rank",            type=int,   default=64,     help="LoRA rank")
    parser.add_argument("--max_length",      type=int,   default=1024,   help="Max token length per example")
    parser.add_argument("--checkpoint_name", type=str,   default="multitask-v1", help="Checkpoint name")
    parser.add_argument("--no_publish",      action="store_true",        help="Skip publishing")
    args = parser.parse_args()

    # Setup
    print(f"Model: {MODEL}")
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

    # Load datasets
    print("\nLoading datasets...")
    math_convos = load_gsm8k()
    if_convos   = load_tulu(N_IF)
    code_convos = load_opencode(N_CODE)

    all_convos = math_convos + if_convos + code_convos
    random.seed(SEED)
    random.shuffle(all_convos)
    print(f"\nTotal conversations: {len(all_convos)}")

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
    tc = sc.create_lora_training_client(base_model=MODEL, rank=args.rank)
    print("  Training client ready")

    # Train
    adam_params = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95, eps=1e-8)
    print(f"\nTraining for {args.num_steps} steps (batch_size={args.batch_size}, lr={args.lr})...")

    for step in range(args.num_steps):
        start = (step * args.batch_size) % len(all_data)
        batch = [all_data[i % len(all_data)] for i in range(start, start + args.batch_size)]

        fwd_bwd_future = tc.forward_backward(batch, loss_fn="cross_entropy")
        optim_future   = tc.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()

        logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
        weights  = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
        loss     = -np.dot(logprobs, weights) / max(weights.sum(), 1)

        if (step + 1) % 50 == 0 or step == 0:
            print(f"  Step {step+1}/{args.num_steps} | Loss: {loss:.4f}")

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
            "num_steps":     args.num_steps,
            "batch_size":    args.batch_size,
            "learning_rate": args.lr,
            "lora_rank":     args.rank,
            "n_math":        len(math_convos),
            "n_if":          len(if_convos),
            "n_code":        len(code_convos),
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