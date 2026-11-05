"""Tests sliding-window attention on a small random model, on the CPU in float32, where rounding is tiny.

usage: test_cache.py <path to ricursive.py>

The reference is one uncached pass, which applies the sliding window through an explicit mask.
"""
import importlib.util
import sys

import mlx.core as mx

mx.set_default_device(mx.cpu)
spec = importlib.util.spec_from_file_location("ricursive_test", sys.argv[1])
ricursive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ricursive)

WINDOW, TOKENS, TOLERANCE = 8, 60, 1e-4
failures = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


mx.random.seed(1)
args = ricursive.ModelArgs(
    model_type="ricursive", hidden_size=64, intermediate_size=128, num_hidden_layers=10,
    num_attention_heads=8, num_key_value_heads=2, head_dim=16, vocab_size=100,
    layer_types=(["sliding_attention"] * 4 + ["full_attention"]) * 2, sliding_window=WINDOW,
    embed_scale=8.0, ffn_clamp_threshold=20.0,
)
model = ricursive.Model(args)
mx.eval(model.parameters())
ids = mx.random.randint(0, 100, (1, TOKENS))
wrong = mx.random.randint(0, 100, (1, TOKENS))
reference = model(ids)


def worst(a, b):
    return mx.abs(a - b).max().item()


for size in (1, 3, 7, 8, 9, 20, 60):
    cache = model.make_cache()
    out = mx.concatenate([model(ids[:, i : i + size], cache=cache) for i in range(0, TOKENS, size)], axis=1)
    gap = worst(reference, out)
    check(f"chunks of {size} match the uncached pass", gap < TOLERANCE, f"{gap:.2e}")

wide = ricursive.Model(ricursive.ModelArgs(**{**args.__dict__, "sliding_window": 100}))
wide.update(model.parameters())
changed = worst(reference, wide(ids))
check("the test can detect a wrong window", changed > 0.01, f"{changed:.3f}")

print("--- rollback, verifying several draft tokens at once", flush=True)
for rejected, drafts in ((1, 3), (3, 3), (4, 4), (2, 4), (15, 15)):
    cache = model.make_cache()
    outs, position = [], 0
    while position < TOKENS:
        keep = drafts - rejected
        row = mx.concatenate([ids[:, position : position + 1 + keep], wrong[:, : rejected]], axis=1)
        real = min(1 + keep, TOKENS - position)
        row = mx.concatenate([ids[:, position : position + real], wrong[:, : rejected]], axis=1)
        out = model(row, cache=cache)
        outs.append(out[:, :real])
        got = {c.trim(rejected) for c in cache}
        if got != {rejected}:
            failures.append("trim")
            print("FAIL trim returned", got, "wanted", rejected, "at", position, flush=True)
            break
        position += real
    out = mx.concatenate(outs, axis=1)
    gap = worst(reference[:, : out.shape[1]], out)
    check(f"rejecting {rejected} of {drafts} drafts on every step, past the window", gap < TOLERANCE and out.shape[1] == TOKENS, f"{gap:.2e}")

cache = model.make_cache()
model(ids, cache=cache)
check("prompt caches can rewind past the window", all(c.is_trimmable() for c in cache))
got = {c.trim(30) for c in cache}
out = model(ids[:, 30:], cache=cache)
gap = worst(reference[:, 30:], out)
check("rewinding 30 tokens and replaying matches", got == {30} and gap < TOLERANCE, f"{gap:.2e}")

print(f"\n{len(failures)} failures: {failures}" if failures else "\nall passed", flush=True)
sys.exit(1 if failures else 0)
