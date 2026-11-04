"""Speed and accuracy of one numeric variant of the MLX port, against reference.npz.

usage: bench_variants.py <checkpoint dir> <reference.npz> <bf16|f16|q8|q6|q4>

Accuracy is scored on the reference's logits: BF16 PyTorch on the same weights.
Quantized variants are built in memory, so nothing is written to disk.
"""

import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step, wired_limit
from mlx_lm.sample_utils import make_sampler

path, ref_path, variant = sys.argv[1:4]
ref = np.load(ref_path)


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def score(name, reference, port):
    p, q = log_softmax(reference.astype(np.float64)), log_softmax(port.astype(np.float64))
    kl = (np.exp(p) * (p - q)).sum(-1)
    top = (reference.argmax(-1) == port.argmax(-1)).mean()
    print(f"  {name:<26} same-top={top * 100:6.2f}% KL mean={kl.mean():.6f} max={kl.max():.6f}", flush=True)


t = time.time()
model, tok = load(path, lazy=True)
if variant == "f16":
    model.set_dtype(mx.float16)
elif variant.startswith("q"):
    nn.quantize(model, group_size=64, bits=int(variant[1:]))
mx.eval(model.parameters())
print(f"[{variant}] ready in {time.time() - t:.0f}s, weights {mx.get_active_memory() / 1e9:.1f} GB", flush=True)


def run(ids, cache=None):
    out = model(mx.array(ids)[None], cache=cache)[0].astype(mx.float32)
    mx.eval(out)
    return np.array(out)


with wired_limit(model):
    score("short prompt", ref["short_logits"], run(ref["short_ids"]))

    cache = model.make_cache()
    long_ids = ref["long_ids"]
    t = time.time()
    chunks = []
    for start in range(0, len(long_ids), 512):
        chunks.append(run(long_ids[start : start + 512], cache))
    prefill = len(long_ids) / (time.time() - t)
    logits = np.concatenate(chunks)
    score("long prompt, last 32", ref["long_tail"], logits[-32:])
    print(f"  long prompt same-top, all 1500: {(ref['long_argmax'] == logits.argmax(-1)).mean() * 100:.2f}%", flush=True)
    print(f"  prefill {prefill:.1f} tok/s", flush=True)
    del cache, chunks, logits

    cache = model.make_cache()
    run(ref["short_ids"], cache)
    steps = np.stack([run(np.array([token]), cache)[-1] for token in ref["greedy"][:-1]])
    score("greedy, reference tokens", ref["greedy_logits"], steps)
    del cache

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Explain what a mutex is in two sentences."}],
        add_generation_prompt=True,
    )
    if not isinstance(prompt, list):
        prompt = list(prompt["input_ids"])
    t0 = time.time()
    first, tokens = None, []
    for token, _ in generate_step(mx.array(prompt), model, max_tokens=48, sampler=make_sampler(temp=0.0)):
        first = first or time.time()
        tokens.append(int(token))
    decode = (len(tokens) - 1) / (time.time() - first)
    print(f"  decode {decode:.2f} tok/s, first token {first - t0:.2f}s", flush=True)
    print(f"  peak MLX memory {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    print("  " + repr(tok.decode(tokens))[:220], flush=True)
