"""Does the drafter change the sampling distribution? Compares repeated short samples with it off and on.

usage:
  sampling_check.py sample <base url> <model> <out.json> [--samples N] [--tokens N] [--temperature T] [--top-p P]
  sampling_check.py compare <off.json> <on.json>

The first token is drawn before any draft exists. Later tokens pass through draft acceptance, so a
faulty acceptance step would show up as different frequencies there. Reports a two-sample chi-square
statistic per position over the paths seen so far, pooling rare ones. A permutation test gives the
p-value.
"""
import argparse
import json
import random
import time
import urllib.request
from collections import Counter

PROMPTS = [
    "Write the opening sentence of a story about a lighthouse.",
    "Name a programming language and say one thing about it.",
    "Give me a random idea for a weekend project.",
]


def sample(args):
    out = {}
    for prompt in PROMPTS:
        rows = []
        started = time.time()
        for _ in range(args.samples):
            body = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.tokens,
                "logprobs": True,
                "stream": False,
            }
            request = urllib.request.Request(
                args.url + "/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                reply = json.load(response)
            rows.append([entry["token"] for entry in reply["choices"][0]["logprobs"]["content"]])
        out[prompt] = rows
        print(f"{len(rows)} samples in {time.time() - started:.0f}s; {prompt[:40]!r}; e.g. {''.join(rows[0])!r}", flush=True)
    json.dump(out, open(args.out, "w"))


def statistic(a, b, keep):
    total = 0.0
    na, nb = len(a), len(b)
    if not na or not nb:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    for token in keep + [None]:
        if token is None:
            x = sum(v for k, v in ca.items() if k not in keep)
            y = sum(v for k, v in cb.items() if k not in keep)
        else:
            x, y = ca[token], cb[token]
        if x + y:
            ex, ey = (x + y) * na / (na + nb), (x + y) * nb / (na + nb)
            total += (x - ex) ** 2 / ex + (y - ey) ** 2 / ey
    return total


def compare(args):
    off, on = json.load(open(args.off)), json.load(open(args.on))
    rng = random.Random(1)
    smallest = 1.0
    tests = 0
    for prompt in off:
        print(prompt[:50])
        width = max(len(r) for r in off[prompt])
        for position in range(width):
            a = ["".join(r[: position + 1]) for r in off[prompt] if len(r) > position]
            b = ["".join(r[: position + 1]) for r in on[prompt] if len(r) > position]
            joined = Counter(a + b)
            keep = [k for k, v in joined.items() if v >= 10]
            if len(keep) < 2:
                print(f"  position {position}: too few repeated paths")
                continue
            seen = statistic(a, b, keep)
            pool = a + b
            greater = 0
            for _ in range(2000):
                rng.shuffle(pool)
                greater += statistic(pool[: len(a)], pool[len(a) :], keep) >= seen
            p = (greater + 1) / 2001
            smallest = min(smallest, p)
            tests += 1
            top = joined.most_common(1)[0][0]
            print(
                f"  position {position}: paths={len(keep)} chi2={seen:.1f} p={p:.3f}; "
                f"top path {top[-24:]!r} off {a.count(top) / len(a):.3f} on {b.count(top) / len(b):.3f}"
            )
    print(f"{tests} tests; smallest p={smallest:.4f}; with no real difference, expect about {1 / max(tests, 1):.3f}")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("sample")
    p.add_argument("url")
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--tokens", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.set_defaults(run=sample)
    p = commands.add_parser("compare")
    p.add_argument("off")
    p.add_argument("on")
    p.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
