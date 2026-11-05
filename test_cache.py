"""Tests SlidingKVCache on a small random model, on the CPU in float32, where rounding is tiny.

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

WINDOW, SPARE, TOKENS, TOLERANCE = 8, 4, 60, 1e-4
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


def make_cache():
    return [
        ricursive.SlidingKVCache(WINDOW, spare=SPARE) if layer.self_attn.is_sliding else ricursive.KVCache()
        for layer in model.layers
    ]


def worst(a, b):
    return mx.abs(a - b).max().item()


for size in (1, 3, 7, 8, 9, 20, 60):
    cache = make_cache()
    out = mx.concatenate([model(ids[:, i : i + size], cache=cache) for i in range(0, TOKENS, size)], axis=1)
    gap = worst(reference, out)
    check(f"chunks of {size} match the uncached pass", gap < TOLERANCE, f"{gap:.2e}")

changed = worst(reference, ricursive.Model(ricursive.ModelArgs(**{**args.__dict__, "sliding_window": 100})).update(model.parameters())(ids))
check("the test can detect a wrong window", changed > 0.01, f"{changed:.3f}")

print("--- rollback, verifying several draft tokens at once", flush=True)
for rejected, drafts in ((1, 3), (3, 3), (4, 4), (2, 4)):
    cache = make_cache()
    outs, position = [], 0
    while position < TOKENS:
        # Verify the next real token and the draft tokens after it, of which some are wrong.
        keep = drafts - rejected
        row = mx.concatenate([ids[:, position : position + 1 + keep], wrong[:, position : position + rejected]], axis=1)
        row = row[:, : TOKENS - position + rejected]
        out = model(row, cache=cache)
        real = min(1 + keep, TOKENS - position)
        outs.append(out[:, :real])
        extra = row.shape[1] - real
        got = [c.trim(extra) for c in cache]
        if any(g != extra for g in got):
            failures.append("trim")
            print("FAIL trim returned", set(got), "wanted", extra, "at", position, flush=True)
            break
        position += real
    out = mx.concatenate(outs, axis=1)
    gap = worst(reference[:, : out.shape[1]], out)
    check(f"rejecting {rejected} of {drafts} drafts on every step, past the window", gap < TOLERANCE and out.shape[1] == TOKENS, f"{gap:.2e}")

print("--- limits", flush=True)
cache = ricursive.SlidingKVCache(WINDOW, spare=SPARE)
k = mx.zeros((1, 2, 1, 16))
for _ in range(5):
    cache.update_and_fetch(k, k)
check("rewindable while the whole history is held", cache.is_trimmable())
for _ in range(40):
    keys, _values = cache.update_and_fetch(k, k)
check("a single token sees exactly the window", keys.shape[2] == WINDOW, str(keys.shape[2]))
check("not rewindable for prompt caches once tokens are dropped", not cache.is_trimmable())
check("memory stays bounded", cache.size() <= WINDOW - 1 + 2 * SPARE + 1, str(cache.size()))
held = cache.size()
trimmed = cache.trim(1000)
check("trim never eats into the window", trimmed == held - (WINDOW - 1) and cache.size() == WINDOW - 1, f"trimmed {trimmed}")
check("trim reports what it did", cache.trim(1) == 0)
check("the offset follows", cache.offset == 45 - trimmed)
state, meta = cache.state, cache.meta_state
copy = ricursive.SlidingKVCache(1)
copy.state, copy.meta_state = state, meta
check("state round-trips", copy.offset == cache.offset and copy.window == WINDOW and copy.size() == cache.size())

print(f"\n{len(failures)} failures: {failures}" if failures else "\nall passed", flush=True)
sys.exit(1 if failures else 0)
