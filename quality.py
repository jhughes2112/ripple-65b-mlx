"""Logit quality of the MLX port, position by position on varied text.

usage:
  quality.py reference <checkpoint> <text dir> <out.npz> --math f32|bf16
      Runs the upstream PyTorch code on the CPU, one uncached pass per segment.
      f32: float32 math on the BF16 weights, which removes BF16's arithmetic rounding.
      Each matrix is widened only while it is used, so memory stays near the BF16 size.
      bf16: the upstream code as shipped, as a yardstick for rounding alone.
  quality.py score <reference.npz> <other.npz>
      Scores saved torch logits against the reference.
  quality.py mlx <checkpoint> <reference.npz> --variant bf16|f16|q8|q6|q4|saved
      Runs a variant through the rotating cache in 512-token chunks, the path a server
      uses, and scores it against the reference. "saved" scores a copy from
      convert_weights.sh without converting it again; "bf16" runs the checkpoint as it is.

Reports the same-top-token rate and KL(reference || run) over the full vocabulary.
"""

import argparse
import json
import pathlib
import time

import numpy as np

SEGMENTS = ["code", "docs", "english", "multilingual", "agent"]
SEG_TOKENS = 1000


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def score(reference, run):
    """Returns the per-position same-top flags and KL values."""
    p = log_softmax(reference.astype(np.float64))
    q = log_softmax(run.astype(np.float64))
    return reference.argmax(-1) == run.argmax(-1), (np.exp(p) * (p - q)).sum(-1)


def report(label, per_segment):
    print(f"[{label}]", flush=True)
    tops, kls = [], []
    for name, (top, kl) in per_segment.items():
        print(
            f"  {name:<14} n={len(kl):<5} same-top={top.mean() * 100:6.2f}% "
            f"KL mean={kl.mean():.6f} p99={np.percentile(kl, 99):.6f} max={kl.max():.6f}",
            flush=True,
        )
        tops.append(top)
        kls.append(kl)
    top, kl = np.concatenate(tops), np.concatenate(kls)
    print(
        f"  {'ALL':<14} n={len(kl):<5} same-top={top.mean() * 100:6.2f}% "
        f"KL mean={kl.mean():.6f} p99={np.percentile(kl, 99):.6f} max={kl.max():.6f}",
        flush=True,
    )


def cmd_reference(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).eval()

    if args.math == "f32":
        original_embedding = torch.nn.Embedding.forward
        torch.nn.Linear.forward = lambda self, x: F.linear(x, self.weight.float(), None)
        torch.nn.Embedding.forward = lambda self, x: original_embedding(self, x).float()

    saved = {}
    with torch.inference_mode():
        for name in SEGMENTS:
            text = (pathlib.Path(args.text_dir) / f"{name}.txt").read_text(encoding="utf-8")
            ids = tok(text, return_tensors="pt").input_ids[:, :SEG_TOKENS]
            t = time.time()
            output = model(ids, use_cache=False).logits[0]
            print(f"{name}: {ids.shape[1]} tokens, {output.dtype}, {time.time() - t:.0f}s", flush=True)
            saved[f"{name}_ids"] = ids[0].numpy()
            saved[f"{name}_logits"] = output.float().numpy()
    np.savez(args.out, **saved)
    print("saved", args.out, flush=True)


def cmd_score(args):
    reference, other = np.load(args.reference), np.load(args.other)
    for name in SEGMENTS:
        assert (reference[f"{name}_ids"] == other[f"{name}_ids"]).all(), name
    report(
        pathlib.Path(args.other).stem,
        {n: score(reference[f"{n}_logits"], other[f"{n}_logits"]) for n in SEGMENTS},
    )


def cmd_mlx(args):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    from mlx_lm.generate import wired_limit

    reference = np.load(args.reference)
    model, _ = load(args.checkpoint, lazy=True)
    if args.variant == "f16":
        model.set_dtype(mx.float16)
    elif args.variant.startswith("q"):
        nn.quantize(model, group_size=64, bits=int(args.variant[1:]))
    mx.eval(model.parameters())

    per_segment = {}
    with wired_limit(model):
        for name in SEGMENTS:
            ids = reference[f"{name}_ids"]
            cache = model.make_cache()
            chunks = []
            for start in range(0, len(ids), 512):
                out = model(mx.array(ids[start : start + 512])[None], cache=cache)[0]
                out = out.astype(mx.float32)
                mx.eval(out)
                chunks.append(np.array(out))
            per_segment[name] = score(reference[f"{name}_logits"], np.concatenate(chunks))
            del cache, chunks
            mx.clear_cache()
    report(f"mlx {args.variant}", per_segment)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("reference")
    p.add_argument("checkpoint")
    p.add_argument("text_dir")
    p.add_argument("out")
    p.add_argument("--math", choices=["f32", "bf16"], required=True)
    p.set_defaults(run=cmd_reference)
    p = commands.add_parser("score")
    p.add_argument("reference")
    p.add_argument("other")
    p.set_defaults(run=cmd_score)
    p = commands.add_parser("mlx")
    p.add_argument("checkpoint")
    p.add_argument("reference")
    p.add_argument("--variant", choices=["bf16", "f16", "q8", "q6", "q4", "saved"], required=True)
    p.set_defaults(run=cmd_mlx)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
