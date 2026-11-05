"""Tests saved samples against the exact next-token probabilities computed from the model.

usage: sampling_truth.py <model dir> <off.json> <on.json> --temperature T --top-p P [--min-count N]

For every prefix seen often enough in both sample sets, computes the true sampling distribution
directly, then measures how far each set's next-token counts are from it. A Monte Carlo draw from
the true distribution gives the p-value, so correlated positions are handled by testing each prefix
against its own truth. If drafter-off fits and drafter-on doesn't, acceptance is at fault.
"""
import argparse
import json
from collections import Counter, defaultdict

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import wired_limit

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("off")
parser.add_argument("on")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top-p", type=float, default=1.0)
parser.add_argument("--min-count", type=int, default=30)
args = parser.parse_args()

off, on = json.load(open(args.off)), json.load(open(args.on))
model, tok = load(args.model)
rng = np.random.default_rng(1)


def sampling_probs(logits):
    """The sampler's distribution: top-p filtered on the original probabilities, then temperature."""
    lp = logits.astype(np.float64)
    lp = lp - lp.max()
    lp = lp - np.log(np.exp(lp).sum())
    if 0 < args.top_p < 1:
        order = np.argsort(lp)
        probs = np.exp(lp[order])
        cumulative = np.cumsum(probs)
        keep = order[cumulative > 1 - args.top_p]
        mask = np.full(lp.shape, -np.inf)
        mask[keep] = lp[keep]
        lp = mask
    scaled = lp / args.temperature
    finite = np.isfinite(scaled)
    scaled = scaled - scaled[finite].max()
    p = np.where(finite, np.exp(np.where(finite, scaled, 0.0)), 0.0)
    return p / p.sum()


def transitions(rows):
    """prefix tuple -> Counter of next token text."""
    table = defaultdict(Counter)
    for row in rows:
        for i in range(1, len(row)):
            table[tuple(row[:i])][row[i]] += 1
    return table


def distance(counts, probs):
    """Chi-square over the categories, rare ones pooled, for a count vector."""
    expected = probs * counts.sum()
    return float((((counts - expected) ** 2) / np.maximum(expected, 1e-12)).sum())


totals = {"off": [], "on": []}
with wired_limit(model):
    for prompt in off:
        print(prompt[:50], flush=True)
        base = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
        base = base if isinstance(base, list) else list(base["input_ids"])
        table_off, table_on = transitions(off[prompt]), transitions(on[prompt])
        nodes = [k for k in table_off if k in table_on and sum(table_off[k].values()) >= args.min_count and sum(table_on[k].values()) >= args.min_count]
        for prefix in sorted(nodes, key=lambda k: (len(k), k)):
            ids = tok.encode("".join(prefix), add_special_tokens=False)
            ids = list(ids) if not isinstance(ids, list) else ids
            if [tok.decode([i]) for i in ids] != list(prefix):
                print(f"  skipped {''.join(prefix)!r}: tokenization differs", flush=True)
                continue
            logits = model(mx.array(base + ids)[None])[0, -1].astype(mx.float32)
            mx.eval(logits)
            probs = sampling_probs(np.array(logits))
            # Categories: tokens that the truth or the samples make common, plus everything else.
            seen = set(table_off[prefix]) | set(table_on[prefix])
            text_to_id = {}
            for candidate in np.argsort(probs)[-400:]:
                text_to_id.setdefault(tok.decode([int(candidate)]), int(candidate))
            common = [t for t in seen if t in text_to_id and probs[text_to_id[t]] * 300 >= 5]
            if len(common) < 1:
                continue
            p = np.array([probs[text_to_id[t]] for t in common])
            p = np.append(p, max(1 - p.sum(), 1e-9))
            p = p / p.sum()
            line = f"  {''.join(prefix)[-26:]!r:<30}"
            for name, table in (("off", table_off), ("on", table_on)):
                n = sum(table[prefix].values())
                counts = np.array([table[prefix][t] for t in common] + [n - sum(table[prefix][t] for t in common)], dtype=float)
                seen_stat = distance(counts, p)
                draws = rng.multinomial(n, p, size=4000).astype(float)
                expected = p * n
                null = (((draws - expected) ** 2) / np.maximum(expected, 1e-12)).sum(axis=1)
                value = float((null >= seen_stat).mean() + 1 / 4001)
                totals[name].append((seen_stat, null))
                top = int(np.argmax(p[:-1]))
                line += f" {name}: n={n:<3} top {counts[top] / n:.3f} p={value:.3f} |"
            top_token = common[int(np.argmax(p[:-1]))]
            print(f"{line} truth {top_token!r} {p[top]:.3f}", flush=True)

print(flush=True)
for name, rows in totals.items():
    observed = sum(r[0] for r in rows)
    pooled = sum(r[1] for r in rows)
    print(f"{name}: {len(rows)} prefixes; total chi2={observed:.1f}; pooled p={(pooled >= observed).mean():.4f}; expected chi2 about {pooled.mean():.1f}", flush=True)
