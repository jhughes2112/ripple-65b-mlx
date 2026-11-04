"""Reference side: Ripple's own PyTorch code on CPU, in BF16. Writes logits for parity_mlx.py.

usage: parity_torch.py <checkpoint dir> <out.npz>

Runs the checkpoint's modeling_ricursive.py, which is code from the model repo
(reviewed at revision 240d5966: torch and transformers only).
"""

import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(path)

SHORT = "Write a Python function that checks whether a string is a palindrome, then explain it."
sentence = (
    "The quick brown fox jumps over the lazy dog while the committee reviews the "
    "quarterly budget, and a small program counts each word it sees. "
)
short_ids = tok(SHORT, return_tensors="pt").input_ids
# Longer than the 1024-token window, so local layers must drop old tokens and
# the global layers must keep them.
long_ids = tok(sentence * 60, return_tensors="pt").input_ids[:, :1500]
print("tokens", short_ids.shape[1], long_ids.shape[1], flush=True)

t = time.time()
model = AutoModelForCausalLM.from_pretrained(
    path, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
).eval()
print(f"loaded in {time.time() - t:.0f}s", flush=True)

with torch.inference_mode():
    t = time.time()
    short_logits = model(short_ids, use_cache=False).logits[0].float().numpy()
    print(f"short pass {time.time() - t:.0f}s", flush=True)

    t = time.time()
    long_logits = model(long_ids, use_cache=False).logits[0].float()
    print(f"long pass {time.time() - t:.0f}s", flush=True)
    long_argmax = long_logits.argmax(-1).numpy()
    long_tail = long_logits[-32:].numpy()

    # Greedy, one token at a time through the cache: the path real generation takes.
    t = time.time()
    output = model(short_ids, use_cache=True)
    cache = output.past_key_values
    next_id = output.logits[0, -1].float().argmax().view(1, 1)
    greedy, greedy_logits = [int(next_id)], []
    position = short_ids.shape[1]
    for _ in range(23):
        output = model(
            next_id,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([position]),
        )
        step = output.logits[0, -1].float()
        greedy_logits.append(step.numpy())
        next_id = step.argmax().view(1, 1)
        greedy.append(int(next_id))
        position += 1
    print(f"greedy {time.time() - t:.0f}s: {tok.decode(greedy)!r}", flush=True)

np.savez(
    out,
    short_ids=short_ids[0].numpy(),
    long_ids=long_ids[0].numpy(),
    short_logits=short_logits,
    long_argmax=long_argmax,
    long_tail=long_tail,
    greedy=np.array(greedy),
    greedy_logits=np.stack(greedy_logits),
)
print("saved", out, flush=True)
