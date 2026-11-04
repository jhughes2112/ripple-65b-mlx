"""Port side: the MLX model against the logits from parity_torch.py.

usage: parity_mlx.py <checkpoint dir> <reference.npz>

Prints same-top-token rates and KL for three paths: a full pass, a long prompt
past the sliding window (both with and without chunked caching), and greedy
one-token decoding through the cache. Then times decoding.
"""

import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

path, ref_path = sys.argv[1], sys.argv[2]
ref = np.load(ref_path)


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def compare(name, reference, port):
    p, q = log_softmax(reference.astype(np.float64)), log_softmax(port.astype(np.float64))
    kl = (np.exp(p) * (p - q)).sum(-1)
    top = (reference.argmax(-1) == port.argmax(-1)).mean()
    print(
        f"{name:<34} positions={len(p):<5} same-top={top * 100:6.2f}% "
        f"KL mean={kl.mean():.6f} max={kl.max():.6f}",
        flush=True,
    )


t = time.time()
model, tok = load(path)
print(f"loaded in {time.time() - t:.0f}s", flush=True)
print("weights dtype", model.lm_head.weight.dtype, flush=True)


def run(ids, cache=None):
    out = model(mx.array(ids)[None], cache=cache)[0].astype(mx.float32)
    mx.eval(out)
    return np.array(out)


short_ids, long_ids = ref["short_ids"], ref["long_ids"]

compare("short prompt, one pass", ref["short_logits"], run(short_ids))

long_logits = run(long_ids)
compare("long prompt, last 32", ref["long_tail"], long_logits[-32:])
top = (ref["long_argmax"] == long_logits.argmax(-1)).mean()
print(f"long prompt, all positions: same-top={top * 100:.2f}%", flush=True)

# The same long prompt through the rotating cache, in chunks.
cache = model.make_cache()
chunks = []
for start in range(0, len(long_ids), 512):
    chunks.append(run(long_ids[start : start + 512], cache))
    mx.eval([c.state for c in cache])
compare("long prompt, chunked + cached", ref["long_tail"], np.concatenate(chunks)[-32:])

# Greedy through the cache, feeding the reference's tokens.
cache = model.make_cache()
logits = run(short_ids, cache)[-1]
ours = [int(logits.argmax())]
steps = []
for token in ref["greedy"][:-1]:
    step = run(np.array([token]), cache)[-1]
    steps.append(step)
    ours.append(int(step.argmax()))
compare("greedy one token at a time", ref["greedy_logits"], np.stack(steps))
print("greedy tokens identical:", bool((np.array(ours) == ref["greedy"]).all()), flush=True)

# Free-running output, then speed.
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Explain what a mutex is in two sentences."}],
    add_generation_prompt=True,
)
sampler = make_sampler(temp=0.0)
t = time.time()
tokens = []
first = None
for token, _ in generate_step(mx.array(prompt), model, max_tokens=120, sampler=sampler):
    if first is None:
        first = time.time()
    tokens.append(int(token))
done = time.time()
print(f"prompt tokens={len(prompt)} first token {first - t:.2f}s", flush=True)
print(f"decode {(len(tokens) - 1) / (done - first):.2f} tok/s over {len(tokens)} tokens", flush=True)
print("peak MLX memory GB", round(mx.get_peak_memory() / 1e9, 1), flush=True)
print(repr(tok.decode(tokens)), flush=True)
