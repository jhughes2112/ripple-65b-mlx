"""mlx-lm's OpenAI-compatible server, optionally quantizing Ripple-65B as it loads.

usage: serve.py [--bits N | --f16] --model <checkpoint dir> [mlx_lm.server arguments]

Converting in memory takes seconds, so no second copy of the weights is needed
on disk. With neither option the model runs on its BF16 weights.
"""

import argparse
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import server


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bits", type=int, choices=[4, 6, 8])
    parser.add_argument("--f16", action="store_true", help="run in float16 instead of BF16")
    args, rest = parser.parse_known_args()

    if args.bits or args.f16:
        original_load = server.load

        def load_variant(*load_args, **load_kwargs):
            model, *others = original_load(*load_args, **{**load_kwargs, "lazy": True})
            if args.f16:
                model.set_dtype(mx.float16)
            else:
                nn.quantize(model, group_size=64, bits=args.bits)
            mx.eval(model.parameters())
            return (model, *others)

        server.load = load_variant

    sys.argv = ["mlx_lm.server", *rest]
    server.main()


if __name__ == "__main__":
    main()
