"""
LoRA checkpoint soup: linearly average the LoRA A/B matrices from multiple
Tinker checkpoints and publish the averaged adapter as a new sampling
checkpoint.

Rationale (Wortsman et al. 2022, "Model Soups"): a simple uniform or
weighted average of fine-tuned weights on the same base often matches or
beats the best individual checkpoint, especially for multi-task settings
where different checkpoints peak on different metrics. In our setup,
IFEval peaks at step 5000 while GSM8K/HumanEval peak at step 8000 -- prime
territory for model soup.

Tinker LoRA checkpoints are opaque tensor bundles; we download them via the
REST client, load with safetensors (tinker serializes adapters that way as
of cookbook 2026-04), average the corresponding keys, and re-upload.

Usage:
    python -m evaluation.improvements.checkpoint_soup \
        --checkpoints tinker://...v5-step5000 tinker://...v5-step8000 \
        --weights 0.5 0.5 \
        --base_model meta-llama/Llama-3.1-8B \
        --name multitask-8b-soup-5k-8k
"""

import argparse
import json
import logging
import os
import tempfile
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _download_checkpoint(rest_client, tinker_path: str, out_dir: str) -> str:
    """Pull the raw adapter file(s) from Tinker into a local directory."""
    import tinker  # lazy import
    logger.info(f"  Downloading {tinker_path} -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    try:
        rest_client.download_checkpoint_to_dir(tinker_path, out_dir).result()
    except AttributeError:
        raise RuntimeError(
            "Your installed tinker REST client does not expose "
            "`download_checkpoint_to_dir`. Fall back to manual download in "
            "the Tinker dashboard and pass --local_paths instead."
        )
    return out_dir


def _load_tensors(path_or_dir: str):
    """Load safetensors / pytorch bin into a dict[str, torch.Tensor]."""
    import torch
    files = []
    if os.path.isdir(path_or_dir):
        for root, _, fnames in os.walk(path_or_dir):
            for fn in fnames:
                if fn.endswith(".safetensors") or fn.endswith(".bin") or fn.endswith(".pt"):
                    files.append(os.path.join(root, fn))
    else:
        files.append(path_or_dir)

    if not files:
        raise RuntimeError(f"No tensor files found under {path_or_dir}")

    tensors = {}
    for fp in files:
        if fp.endswith(".safetensors"):
            from safetensors.torch import load_file
            part = load_file(fp)
        else:
            part = torch.load(fp, map_location="cpu")
        overlap = set(tensors) & set(part)
        if overlap:
            logger.warning(f"  Overlapping keys ({len(overlap)}) across shard files; later shards overwrite.")
        tensors.update(part)
    return tensors


def _save_tensors(tensors, out_dir: str):
    """Write a single safetensors file into `out_dir`."""
    import torch  # noqa: F401
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "adapter.safetensors")
    save_file(tensors, out_path)
    logger.info(f"  Wrote averaged adapter to {out_path}")
    return out_path


def average_tensors(tensor_dicts, weights):
    """
    Linear combination of matching keys.
    Only averages float tensors; for non-float keys (metadata ints etc.)
    we keep the value from the first dict.
    """
    import torch
    assert len(tensor_dicts) == len(weights)
    weights = [float(w) for w in weights]
    s = sum(weights)
    if s == 0:
        raise ValueError("Weights sum to zero.")
    weights = [w / s for w in weights]

    ref_keys = set(tensor_dicts[0].keys())
    for i, td in enumerate(tensor_dicts[1:], 1):
        missing = ref_keys - set(td.keys())
        extra = set(td.keys()) - ref_keys
        if missing or extra:
            logger.warning(
                f"  Checkpoint {i} key mismatch: {len(missing)} missing, {len(extra)} extra. "
                f"Using intersection."
            )
            ref_keys &= set(td.keys())

    averaged = {}
    for k in ref_keys:
        t0 = tensor_dicts[0][k]
        if not t0.is_floating_point():
            averaged[k] = t0
            continue
        acc = torch.zeros_like(t0, dtype=torch.float32)
        for w, td in zip(weights, tensor_dicts):
            acc += w * td[k].to(torch.float32)
        averaged[k] = acc.to(t0.dtype)
    return averaged


def main():
    p = argparse.ArgumentParser(description="Average LoRA adapter weights across checkpoints")
    p.add_argument("--checkpoints", nargs="+", default=None,
                   help="Tinker checkpoint paths (tinker://...) to average")
    p.add_argument("--local_paths", nargs="+", default=None,
                   help="Already-downloaded directories or files (skip download)")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Mixing weights (default: uniform)")
    p.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--name", type=str, default="multitask-soup")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to cache averaged adapter (default: improvements_results/soup)")
    p.add_argument("--publish", action="store_true",
                   help="Upload averaged adapter to Tinker and publish as a sampler checkpoint")
    p.add_argument("--dry_run", action="store_true",
                   help="Average locally but skip upload")
    args = p.parse_args()

    if not args.checkpoints and not args.local_paths:
        p.error("Provide --checkpoints or --local_paths.")
    if args.checkpoints and args.local_paths:
        p.error("Use either --checkpoints or --local_paths, not both.")

    n = len(args.checkpoints or args.local_paths)
    weights = args.weights or [1.0] * n
    if len(weights) != n:
        p.error(f"Got {n} checkpoints but {len(weights)} weights.")

    out_root = args.output_dir or os.path.join(EVAL_DIR, "improvements_results", "soup", args.name)
    os.makedirs(out_root, exist_ok=True)

    # Step 1: obtain local tensors.
    tensor_dicts = []
    if args.local_paths:
        for lp in args.local_paths:
            tensor_dicts.append(_load_tensors(lp))
    else:
        import tinker
        sc = tinker.ServiceClient()
        rest_client = sc.create_rest_client()
        for i, ckpt in enumerate(args.checkpoints):
            local = os.path.join(out_root, f"ckpt_{i}")
            _download_checkpoint(rest_client, ckpt, local)
            tensor_dicts.append(_load_tensors(local))

    # Step 2: average.
    logger.info(f"Averaging {n} adapters with weights {weights}")
    averaged = average_tensors(tensor_dicts, weights)

    avg_dir = os.path.join(out_root, "averaged")
    avg_path = _save_tensors(averaged, avg_dir)

    info = {
        "name": args.name,
        "base_model": args.base_model,
        "sources": args.checkpoints or args.local_paths,
        "weights": weights,
        "averaged_path": avg_path,
        "n_keys": len(averaged),
    }

    # Step 3: optionally upload + publish.
    if args.publish and not args.dry_run:
        try:
            import tinker
            sc = tinker.ServiceClient()
            rest_client = sc.create_rest_client()
            logger.info(f"Uploading {avg_path} ...")
            # NOTE: Tinker's exact upload method for adapter weights evolves.
            # The common paths are `upload_adapter_from_dir` or
            # `create_checkpoint_from_dir`. Wrap in getattr for forward-compat.
            upload_fn = (
                getattr(rest_client, "upload_adapter_from_dir", None)
                or getattr(rest_client, "create_checkpoint_from_dir", None)
            )
            if upload_fn is None:
                raise RuntimeError(
                    "Tinker REST client has no known upload method. "
                    "Upload manually via the dashboard."
                )
            new_path = upload_fn(avg_dir, base_model=args.base_model, name=args.name).result()
            info["published_path"] = str(new_path)
            logger.info(f"Published averaged adapter: {new_path}")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            info["upload_error"] = str(e)

    info_path = os.path.join(out_root, "soup_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Wrote {info_path}")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
