"""Writes a converted copy of Ripple-65B that mlx-lm can load directly.

usage: convert.py <checkpoint dir> <output dir> f16|q8|q6|q4

Applies the same conversion as serve.py and quality.py.
"""

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load, save

GROUP_SIZE = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    parser.add_argument("variant", choices=["f16", "q8", "q6", "q4"])
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")

    model, tokenizer, config = load(args.checkpoint, lazy=True, return_config=True)
    if args.variant == "f16":
        model.set_dtype(mx.float16)
    else:
        bits = int(args.variant[1:])
        nn.quantize(model, group_size=GROUP_SIZE, bits=bits)
        config["quantization"] = {"group_size": GROUP_SIZE, "bits": bits, "mode": "affine"}

    # The output runs on the MLX port, not the checkpoint's PyTorch code.
    config.pop("auto_map", None)
    save(out, args.checkpoint, model, tokenizer, config)
    for path in out.glob("*_ricursive.py"):
        path.unlink()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
