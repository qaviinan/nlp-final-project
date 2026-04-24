"""
<<<<<<< HEAD
GRPO (Group Relative Policy Optimization) fine-tuning for IFEval.
Starts from a pre-trained SFT checkpoint (v4) and applies RL using
IFEval constraint satisfaction as the reward signal.

How it works:
  1. Sample a prompt from argilla ifeval-like data
  2. Generate G=8 outputs using the current model
  3. Score each output with the IFEval constraint checker (0 or 1)
  4. Compute group-relative advantages: A_i = (r_i - mean(r)) / (std(r) + eps)
  5. Train on outputs with positive advantage, weighted by advantage magnitude
  6. Repeat

This is a simplified GRPO — we use rejection sampling (skip negative advantages)
rather than a full clipped policy gradient, which is implementable with Tinker's
forward_backward + weighted datums.

Usage:
    python evaluation/grpo_train.py --sft_checkpoint "tinker://..."
    python evaluation/grpo_train.py --sft_checkpoint "tinker://..." --num_steps 300
=======
GRPO fine-tuning for IFEval.
Starts from a pre-trained SFT checkpoint and applies RL using
IFEval constraint satisfaction as the reward signal.

Usage:
    python -m evaluation.grpo_train \
        --sft_checkpoint "tinker://...weights/..." \
        --sampler_checkpoint "tinker://...sampler_weights/..."
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
"""

import argparse
import json
import os
import random
<<<<<<< HEAD
import re
=======
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430

import numpy as np
import tinker
from datasets import load_dataset
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

<<<<<<< HEAD
# IFEval reward function uses instructions_registry directly

=======
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
MODEL = "meta-llama/Llama-3.1-8B"
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

<<<<<<< HEAD
# GRPO hyperparameters
G = 16          # Number of samples per prompt (increased for better advantage estimation)
EPS = 1e-6      # Advantage normalization epsilon
TEMP = 0.6      # Sampling temperature — lower for focused exploration at high performance


def load_ifeval_prompts(n=2000):
    """Load prompts with constraint metadata from argilla ifeval-like-data."""
    print(f"  Loading argilla IFEval prompts (up to {n})...")
=======
G = 16
EPS = 1e-6
TEMP = 1.0
STOP_SEQUENCES = ["\nUser:", "\nuser:", "User:", "user:"]


def load_ifeval_prompts(n=2000, min_constraints=2):
    """Load prompts with multiple constraints for better GRPO variance."""
    print(f"  Loading argilla IFEval prompts (up to {n}, min_constraints={min_constraints})...")
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
    ds = load_dataset("argilla/ifeval-like-data", "filtered", split="train")
    prompts = []
    for ex in ds:
        if not ex.get("prompt_level_strict_acc", False):
            continue
<<<<<<< HEAD
        prompts.append({
            "prompt": ex["prompt"],
            "instruction_id_list": ex["instruction_id_list"],
=======
        # Only use prompts with multiple constraints — more likely to get variance
        instruction_ids = ex["instruction_id_list"]
        if len(instruction_ids) < min_constraints:
            continue
        prompts.append({
            "prompt": ex["prompt"],
            "instruction_id_list": instruction_ids,
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
            "kwargs_list": json.loads(ex["kwargs"]) if isinstance(ex["kwargs"], str) else ex["kwargs"],
        })
        if len(prompts) >= n:
            break
    random.seed(SEED)
    random.shuffle(prompts)
    print(f"  Loaded {len(prompts)} prompts")
    return prompts


def score_ifeval_response(prompt_text, response_text, instruction_id_list, kwargs_list):
    """
<<<<<<< HEAD
    Score a response against IFEval constraints.
    Returns fraction of constraints passing (partial credit).
    e.g. 2/3 constraints passing = 0.667 reward.
    This gives dense gradient signal vs binary 0/1 scoring.
=======
    Partial credit reward: fraction of constraints passing.
    e.g. 2/3 constraints = 0.667 reward.
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
    """
    try:
        from instruction_following_eval.instructions_registry import INSTRUCTION_DICT

<<<<<<< HEAD
        # Parse kwargs if string
        if isinstance(kwargs_list, str):
            kwargs_list = json.loads(kwargs_list)

        # kwargs_list is a list aligned with instruction_id_list
        # Each element is the kwargs dict for the corresponding instruction
=======
        if isinstance(kwargs_list, str):
            kwargs_list = json.loads(kwargs_list)

>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
        if not isinstance(kwargs_list, list):
            kwargs_list = [{}] * len(instruction_id_list)

        results = []
        for i, instruction_id in enumerate(instruction_id_list):
            if instruction_id not in INSTRUCTION_DICT:
                continue
<<<<<<< HEAD
            # Get per-instruction kwargs (not merged — avoids key collisions)
=======
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
            inst_kwargs = kwargs_list[i] if i < len(kwargs_list) else {}
            if not isinstance(inst_kwargs, dict):
                inst_kwargs = {}
            instruction_cls = INSTRUCTION_DICT[instruction_id]
            try:
                instruction = instruction_cls(instruction_id)
                instruction.build_description(**{
                    k: v for k, v in inst_kwargs.items() if v is not None
                })
                results.append(float(instruction.check_following(response_text)))
            except Exception:
                results.append(0.0)

        if not results:
            return 0.0
<<<<<<< HEAD
        # Partial credit: fraction of constraints passing
=======
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
        return sum(results) / len(results)

    except Exception:
        return 0.0


<<<<<<< HEAD
def render_prompt(prompt_text, renderer, tokenizer):
    """Convert a plain prompt string to tokenized ModelInput.
    build_generation_prompt returns a ModelInput directly."""
    convo = [{"role": "user", "content": prompt_text}]
    return renderer.build_generation_prompt(convo)


def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for IFEval")
    parser.add_argument("--sft_checkpoint", type=str, required=True,
                        help="Tinker training state path from checkpoint_info.json (training_state_path)")
    parser.add_argument("--sampler_checkpoint", type=str, required=True,
                        help="Tinker sampler path from checkpoint_info.json (checkpoint_path)")
    parser.add_argument("--num_steps",       type=int,   default=600)
    parser.add_argument("--lr",              type=float, default=1e-5,
                        help="Lower LR than SFT — RL updates are noisier")
    parser.add_argument("--warmup_steps",    type=int,   default=30)
    parser.add_argument("--weight_decay",    type=float, default=0.01)
    parser.add_argument("--grad_clip_norm",  type=float, default=1.0)
    parser.add_argument("--rank",            type=int,   default=128)
    parser.add_argument("--g_samples",       type=int,   default=G,  # 16
                        help="Number of outputs to sample per prompt")
    parser.add_argument("--max_new_tokens",  type=int,   default=512)
    parser.add_argument("--checkpoint_name", type=str,   default="multitask-8b-v4-grpo-v3")
    parser.add_argument("--no_publish",      action="store_true")
    args = parser.parse_args()

    # Setup
    print(f"Model: {MODEL}")
    print(f"Starting from SFT checkpoint: {args.sft_checkpoint}")
=======
def main():
    parser = argparse.ArgumentParser(description="GRPO fine-tuning for IFEval")
    parser.add_argument("--sft_checkpoint",     type=str, required=True)
    parser.add_argument("--sampler_checkpoint", type=str, required=True)
    parser.add_argument("--num_steps",          type=int,   default=600)
    parser.add_argument("--lr",                 type=float, default=1e-5)
    parser.add_argument("--warmup_steps",       type=int,   default=30)
    parser.add_argument("--weight_decay",       type=float, default=0.01)
    parser.add_argument("--grad_clip_norm",     type=float, default=1.0)
    parser.add_argument("--g_samples",          type=int,   default=G)
    parser.add_argument("--max_new_tokens",     type=int,   default=512)
    parser.add_argument("--checkpoint_name",    type=str,   default="multitask-8b-grpo")
    parser.add_argument("--no_publish",         action="store_true")
    args = parser.parse_args()

    print(f"Model: {MODEL}")
    print(f"Starting from: {args.sft_checkpoint}")
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
    tokenizer = get_tokenizer(MODEL)
    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"Renderer: {renderer_name}")

<<<<<<< HEAD
    # Load prompts
    print("\nLoading IFEval prompts...")
    prompts = load_ifeval_prompts(n=2000)

    # Initialize clients
    print("\nInitializing Tinker clients...")
    sc = tinker.ServiceClient()

    # Training client — start from SFT checkpoint
    tc = sc.create_training_client_from_state(args.sft_checkpoint)
    print("  Training client loaded from SFT checkpoint")

    # Sampling client — uses sampler checkpoint for generation
=======
    print("\nLoading IFEval prompts...")
    prompts = load_ifeval_prompts(n=2000)

    print("\nInitializing Tinker clients...")
    sc = tinker.ServiceClient()
    tc = sc.create_training_client_from_state(args.sft_checkpoint)
    print("  Training client loaded")

>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
    sampling_client = sc.create_sampling_client(model_path=args.sampler_checkpoint)
    print("  Sampling client ready")

    sampling_params = types.SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=TEMP,
        top_p=0.95,
<<<<<<< HEAD
    )

    # GRPO training loop
    print(f"\nStarting GRPO for {args.num_steps} steps "
          f"(G={args.g_samples} samples/prompt, lr={args.lr})...")

    step_rewards = []
    step_losses = []
    skipped_steps = 0  # Steps where all rewards identical (no learning signal)

    for step in range(args.num_steps):
        # Pick a prompt
=======
        stop=STOP_SEQUENCES,
    )

    print(f"\nStarting GRPO for {args.num_steps} steps "
          f"(G={args.g_samples}, lr={args.lr}, temp={TEMP})...")

    step_rewards = []
    step_losses = []
    skipped_steps = 0

    for step in range(args.num_steps):
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
        prompt_data = prompts[step % len(prompts)]
        prompt_text = prompt_data["prompt"]
        instruction_ids = prompt_data["instruction_id_list"]
        kwargs_list = prompt_data["kwargs_list"]

<<<<<<< HEAD
        # Render prompt to tokens
        try:
            model_input = render_prompt(prompt_text, renderer, tokenizer)
=======
        try:
            model_input = renderer.build_generation_prompt(
                [{"role": "user", "content": prompt_text}]
            )
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
        except Exception as e:
            print(f"  Step {step+1}: prompt render failed ({e}), skipping")
            skipped_steps += 1
            continue

<<<<<<< HEAD
        # Sample G outputs
=======
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
        try:
            sample_future = sampling_client.sample(
                num_samples=args.g_samples,
                prompt=model_input,
                sampling_params=sampling_params,
            )
            sample_result = sample_future.result()
            responses = [tokenizer.decode(s.tokens) for s in sample_result.sequences]
        except Exception as e:
            print(f"  Step {step+1}: sampling failed ({e}), skipping")
            skipped_steps += 1
            continue

<<<<<<< HEAD
        # Score each response
=======
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
        rewards = np.array([
            score_ifeval_response(prompt_text, r, instruction_ids, kwargs_list)
            for r in responses
        ])

        mean_reward = rewards.mean()
        step_rewards.append(mean_reward)

<<<<<<< HEAD
        # Skip if all rewards identical — no learning signal
        if rewards.std() < EPS:
            skipped_steps += 1
            if (step + 1) % 50 == 0 or step == 0:
                print(f"  Step {step+1}/{args.num_steps} | "
                      f"Reward: {mean_reward:.3f} | No variance, skipped")
            continue

        # Compute group-relative advantages
        advantages = (rewards - mean_reward) / (rewards.std() + EPS)

        # Build weighted training datums for positive-advantage outputs only
        train_datums = []
        for response_text, advantage in zip(responses, advantages):
            if advantage <= 0:
                continue  # Skip negative/zero advantage outputs
=======
        if rewards.std() < EPS:
            skipped_steps += 1
            print(f"  Step {step+1}/{args.num_steps} | "
                  f"Reward: {mean_reward:.3f} | No variance, skipped")
            continue

        advantages = (rewards - mean_reward) / (rewards.std() + EPS)

        train_datums = []
        for response_text, advantage in zip(responses, advantages):
            if advantage <= 0:
                continue
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
            try:
                convo = [
                    {"role": "user",      "content": prompt_text},
                    {"role": "assistant", "content": response_text},
                ]
                datum = conversation_to_datum(
                    convo,
                    renderer,
                    max_length=1024,
                    train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
                )
<<<<<<< HEAD
                # Scale token weights by advantage
=======
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
                datum.loss_fn_inputs["weights"] *= float(advantage)
                train_datums.append(datum)
            except Exception:
                continue

        if not train_datums:
            skipped_steps += 1
            continue

<<<<<<< HEAD
        # LR warmup
=======
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
        current_lr = args.lr * min(1.0, (step + 1) / max(args.warmup_steps, 1))

        adam_params = types.AdamParams(
            learning_rate=current_lr,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
        )

<<<<<<< HEAD
        # Gradient update
        try:
            fwd_bwd_future = tc.forward_backward(train_datums, loss_fn="cross_entropy")
            optim_future   = tc.optim_step(adam_params)

=======
        try:
            fwd_bwd_future = tc.forward_backward(train_datums, loss_fn="cross_entropy")
            optim_future   = tc.optim_step(adam_params)
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            logprobs = np.concatenate([o["logprobs"].tolist() for o in fwd_bwd_result.loss_fn_outputs])
            weights  = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in train_datums])
            loss     = -np.dot(logprobs, weights) / max(weights.sum(), 1)
            step_losses.append(loss)
        except Exception as e:
            print(f"  Step {step+1}: gradient update failed ({e}), skipping")
            skipped_steps += 1
            continue

        if (step + 1) % 50 == 0 or step == 0:
            avg_reward = np.mean(step_rewards[-50:])
            avg_loss   = np.mean(step_losses[-50:]) if step_losses else 0.0
            print(f"  Step {step+1}/{args.num_steps} | "
                  f"Reward: {mean_reward:.3f} | Avg(50): {avg_reward:.3f} | "
<<<<<<< HEAD
                  f"Loss: {loss:.4f} | Skipped: {skipped_steps} | LR: {current_lr:.2e}")

    # Save checkpoint
    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Checkpoint saved: {checkpoint_path}")

    # Publish
=======
                  f"Loss: {avg_loss:.4f} | Skipped: {skipped_steps} | LR: {current_lr:.2e}")

    print(f"\nSaving checkpoint '{args.checkpoint_name}'...")
    state = tc.save_state(name=args.checkpoint_name).result()
    training_state_path = state.path
    ckpt = tc.save_weights_for_sampler(name=args.checkpoint_name).result()
    checkpoint_path = ckpt.path
    print(f"  Training state: {training_state_path}")
    print(f"  Sampler checkpoint: {checkpoint_path}")

>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
    if not args.no_publish:
        print("\nPublishing checkpoint...")
        rest_client = sc.create_rest_client()
        rest_client.publish_checkpoint_from_tinker_path(checkpoint_path).result()
<<<<<<< HEAD
        print("  Published successfully!")

    # Save info
    info = {
        "checkpoint_path": checkpoint_path,
        "sft_checkpoint":  args.sft_checkpoint,
        "base_model":      MODEL,
        "method":          "GRPO",
        "training": {
            "num_steps":        args.num_steps,
            "g_samples":        args.g_samples,
            "learning_rate":    args.lr,
            "warmup_steps":     args.warmup_steps,
            "weight_decay":     args.weight_decay,
            "grad_clip_norm":   args.grad_clip_norm,
            "temperature":      TEMP,
            "task":             "IFEval",
            "reward_fn":        "instruction_following_eval strict",
            "skipped_steps":    skipped_steps,
=======
        print("  Published!")

    info = {
        "checkpoint_path":     checkpoint_path,
        "training_state_path": training_state_path,
        "sft_checkpoint":      args.sft_checkpoint,
        "sampler_checkpoint":  args.sampler_checkpoint,
        "base_model":          MODEL,
        "method":              "GRPO",
        "training": {
            "num_steps":       args.num_steps,
            "g_samples":       args.g_samples,
            "learning_rate":   args.lr,
            "warmup_steps":    args.warmup_steps,
            "weight_decay":    args.weight_decay,
            "grad_clip_norm":  args.grad_clip_norm,
            "temperature":     TEMP,
            "task":            "IFEval",
            "reward_fn":       "partial credit fraction of constraints passing",
            "skipped_steps":   skipped_steps,
>>>>>>> 83069257d78942438deb3635fdf3b9a3fa560430
            "final_avg_reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
        },
        "published": not args.no_publish,
    }
    info_path = os.path.join(EVAL_DIR, "grpo_checkpoint_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nCheckpoint info saved to {info_path}")
    print(f"\nNext: evaluate your checkpoint with")
    print(f'  python -m evaluation.eval_all --checkpoint_path "{checkpoint_path}" --base_model {MODEL}')


if __name__ == "__main__":
    main()