"""Checks speculative decoding with Ripple's DFlash2 drafter, using oMLX's drafter on the MLX port.

usage:
  dflash_check.py forced <checkpoint> <drafter> <reference.npz> [options]
      Follows each reference text with real drafts and real rollbacks. Every logit that gets committed
      is scored against the reference, so the rollback and batching are tested at every position.
      Also reports how well the drafter predicts the target's top token.
  dflash_check.py greedy <checkpoint> <drafter> <reference.npz> [options]
      Generates freely from a prompt taken from each text, once token by token and once with the
      drafter. The tokens must be identical, and the logits are compared at every position.

Needs an environment with oMLX, mlx-lm, and ricursive.py linked into mlx-lm.
"""

import argparse
import json
import time
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.generate import wired_limit

from omlx.patches.mlx_lm_mtp import drafter_target
from omlx.speculative import dflash as dd


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def score(reference, run):
    p, q = log_softmax(reference.astype(np.float64)), log_softmax(run.astype(np.float64))
    return reference.argmax(-1) == run.argmax(-1), (np.exp(p) * (p - q)).sum(-1)


def summary(top, kl):
    return (
        f"same-top={top.mean() * 100:6.2f}% KL mean={kl.mean():.6f} "
        f"p99={np.percentile(kl, 99):.6f} max={kl.max():.6f} n={len(kl)}"
    )


def setup(args):
    model, tok = load(args.checkpoint, lazy=True)
    if args.variant == "f16":
        model.set_dtype(mx.float16)
    elif args.variant.startswith("q"):
        nn.quantize(model, group_size=64, bits=int(args.variant[1:]))
    mx.eval(model.parameters())
    drafter_target.apply(model)
    drafter = dd.load_dflash_drafter(args.drafter, model, block_size=args.depth + 1, verify_mode="off")
    if args.embed_scale == "one":
        drafter.model.embed_scale = 1.0
    drafter.query_windows = args.anchor == "query"
    if args.no_nope:
        drafter.no_rope_layers = frozenset()
    if args.one_window:
        drafter.layer_windows = None
    print(
        f"[{args.variant}] depth={args.depth} embed_scale={drafter.model.embed_scale} anchor={args.anchor} "
        f"window={drafter.window} layers={drafter.target_layer_ids} no_rope={sorted(drafter.no_rope_layers)} "
        f"layer_windows={drafter.layer_windows}",
        flush=True,
    )
    return model, tok, drafter


class Run:
    """One sequence: prefill, then draft and verify."""

    def __init__(self, model, drafter, uid):
        self.model, self.drafter = model, drafter
        self.cache = model.make_cache()
        self.ids = drafter.target_layer_ids
        self.state = SimpleNamespace(uid=uid, depth=drafter.depth, controller=None, stats=None)
        self.pending = None

    def forward(self, tokens):
        out = self.model(mx.array(tokens)[None], cache=self.cache, capture_layer_ids=self.ids)
        logits = out.logits[0].astype(mx.float32)
        mx.eval(logits, out.hidden_states, [c.state for c in self.cache])
        return np.array(logits), out.hidden_states

    def prefill(self, tokens):
        logits, captures = [], []
        for start in range(0, len(tokens), 512):
            row, hidden = self.forward(tokens[start : start + 512])
            logits.append(row)
            captures.append(hidden)
        self.pending = [mx.concatenate([c[i] for c in captures], axis=1) for i in range(len(self.ids))]
        mx.eval(self.pending)
        return np.concatenate(logits)

    def draft(self, anchor, count):
        self.drafter.draft([(None, self.state, self.pending, mx.array([int(anchor)]), None)])
        mx.eval(self.state.drafts)
        return [int(t) for t in self.state.drafts.tolist()][:count]

    def verify(self, anchor, drafts):
        return self.forward([int(anchor)] + drafts)

    def commit(self, hidden, kept, total):
        self.pending = [c[:, :kept] for c in hidden]
        mx.eval(self.pending)
        rejected = total - kept
        if rejected and not self.model.mtp_partial_rollback(self.cache, 0, rejected):
            raise RuntimeError(f"rollback of {rejected} failed")


def segments(reference, wanted):
    names = [k[: -len("_ids")] for k in reference.files if k.endswith("_ids")]
    return [n for n in names if not wanted or n in wanted]


def cmd_forced(args):
    reference = np.load(args.reference)
    model, _tok, drafter = setup(args)
    tops, kls = [], []
    by_position = np.zeros((args.depth, 2), dtype=np.int64)
    verifies = committed = 0
    started = time.time()
    with wired_limit(model):
        for uid, name in enumerate(segments(reference, args.segments)):
            ids = [int(t) for t in reference[f"{name}_ids"]]
            if args.limit:
                ids = ids[: args.limit]
            run = Run(model, drafter, uid)
            logits = [run.prefill(ids[: args.prefill])]
            position = args.prefill
            seg_verifies = seg_committed = 0
            while position < len(ids):
                # Real drafts, wrong ones included, are verified and rolled back.
                room = len(ids) - position - 1
                drafts = run.draft(ids[position], args.depth)[: room]
                row, hidden = run.verify(ids[position], drafts)
                accepted = 0
                while accepted < len(drafts) and drafts[accepted] == ids[position + 1 + accepted]:
                    accepted += 1
                kept = accepted + 1
                logits.append(row[:kept])
                top = row.argmax(-1)
                for j, draft in enumerate(drafts):
                    if j >= kept:
                        break
                    by_position[j, 0] += 1
                    by_position[j, 1] += int(draft == int(top[j]))
                run.commit(hidden, kept, len(drafts) + 1)
                position += kept
                seg_verifies += 1
                seg_committed += kept
            out = np.concatenate(logits)
            top, kl = score(reference[f"{name}_logits"][: len(out)], out)
            # The same positions without any drafting: the cost of speculation is the difference.
            cache = model.make_cache()
            plain = []
            for start in range(0, len(ids), 512):
                row = model(mx.array(ids[start : start + 512])[None], cache=cache)[0].astype(mx.float32)
                mx.eval(row, [c.state for c in cache])
                plain.append(np.array(row))
            plain = np.concatenate(plain)
            region = args.prefill
            truth = reference[f"{name}_logits"][region : len(out)]
            print(f"  {name:<12} drafted positions, speculative vs reference: {summary(*score(truth, out[region:]))}", flush=True)
            print(f"  {name:<12} drafted positions, plain vs reference:       {summary(*score(truth, plain[region : len(out)]))}", flush=True)
            print(f"  {name:<12} drafted positions, speculative vs plain:     {summary(*score(plain[region : len(out)], out[region:]))}", flush=True)
            del cache, plain
            tops.append(top)
            kls.append(kl)
            print(f"  {name:<12} {summary(top, kl)}; past first 1024: {summary(*[x[1024:] for x in (top, kl)]) if len(kl) > 1024 else 'n/a'}", flush=True)
            print(f"  {name:<12} {seg_committed / max(seg_verifies, 1):.2f} tokens per verify over {seg_verifies} verifies", flush=True)
            verifies += seg_verifies
            committed += seg_committed
            mx.clear_cache()
    print(f"  ALL          {summary(np.concatenate(tops), np.concatenate(kls))}", flush=True)
    rates = [f"{h / n * 100:.0f}%" if n else "-" for n, h in by_position]
    print(f"  draft matches the target's top token, by depth: {' '.join(rates)}", flush=True)
    print(f"  {committed / max(verifies, 1):.2f} text tokens per verify; {time.time() - started:.0f}s; peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)


def cmd_accept(args):
    """Acceptance on a text longer than the reference, to reach past the draft's sliding window."""
    model, tok, drafter = setup(args)
    ids = tok.encode(open(args.reference, encoding="utf-8").read())
    if not isinstance(ids, list):
        ids = list(ids["input_ids"])
    ids = ids[: args.limit]
    print(f"  {len(ids)} tokens, drafting from {args.prefill}", flush=True)
    by_position = np.zeros((args.depth, 2), dtype=np.int64)
    verifies = committed = 0
    started = time.time()
    with wired_limit(model):
        run = Run(model, drafter, 0)
        run.prefill(ids[: args.prefill])
        print(f"  prefilled in {time.time() - started:.0f}s", flush=True)
        started = time.time()
        position = args.prefill
        while position < len(ids):
            drafts = run.draft(ids[position], args.depth)[: len(ids) - position - 1]
            row, hidden = run.verify(ids[position], drafts)
            top = row.argmax(-1)
            accepted = 0
            while accepted < len(drafts) and drafts[accepted] == ids[position + 1 + accepted]:
                accepted += 1
            kept = accepted + 1
            for j, draft in enumerate(drafts[:kept]):
                by_position[j, 0] += 1
                by_position[j, 1] += int(draft == int(top[j]))
            run.commit(hidden, kept, len(drafts) + 1)
            position += kept
            verifies += 1
            committed += kept
    rates = [f"{h / n * 100:.0f}%" if n else "-" for n, h in by_position]
    print(f"  draft matches the target's top token, by depth: {' '.join(rates)}", flush=True)
    print(f"  {committed / max(verifies, 1):.3f} tokens per verify over {verifies}; {time.time() - started:.0f}s", flush=True)


CHAT_PROMPTS = {
    "chat-code": "Write a Python class for an LRU cache with get and put, plus a short usage example.",
    "chat-explain": "Explain how a B-tree index works and why databases use one instead of a hash table.",
    "chat-math": "A tank holds 500 liters and drains at 12 liters per minute while being refilled at 7. How long until it is empty? Show your steps.",
}


def plain_greedy(model, prompt, count):
    cache = model.make_cache()
    logits = []
    for start in range(0, len(prompt), 512):
        out = model(mx.array(prompt[start : start + 512])[None], cache=cache)[0, -1].astype(mx.float32)
        mx.eval(out, [c.state for c in cache])
    row = np.array(out)
    tokens = []
    t = time.time()
    for _ in range(count):
        token = int(row.argmax())
        tokens.append(token)
        logits.append(row)
        out = model(mx.array([token])[None], cache=cache)[0, -1].astype(mx.float32)
        mx.eval(out, [c.state for c in cache])
        row = np.array(out)
    return tokens, np.stack(logits), time.time() - t


def speculative_greedy(model, drafter, prompt, count, uid):
    run = Run(model, drafter, uid)
    row = run.prefill(prompt[:-1])
    # The prompt's last token is the first anchor, so nothing about it is special to the plain run.
    anchor, tokens, logits = prompt[-1], [], []
    verifies = 0
    t = time.time()
    while len(tokens) < count:
        drafts = run.draft(anchor, drafter.depth)[: count - len(tokens) - 1]
        row, hidden = run.verify(anchor, drafts)
        top = row.argmax(-1)
        accepted = 0
        while accepted < len(drafts) and drafts[accepted] == int(top[accepted]):
            accepted += 1
        kept = accepted + 1
        for j in range(kept):
            logits.append(row[j])
            tokens.append(int(top[j]))
        anchor = tokens[-1]
        run.commit(hidden, kept, len(drafts) + 1)
        verifies += 1
    return tokens, np.stack(logits), verifies, time.time() - t


def cmd_greedy(args):
    reference = np.load(args.reference)
    model, tok, drafter = setup(args)
    prompts = {name: [int(t) for t in reference[f"{name}_ids"]][: args.prefill] for name in segments(reference, args.segments)}
    if args.chat:
        for name, text in CHAT_PROMPTS.items():
            prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
            prompts[name] = prompt if isinstance(prompt, list) else list(prompt["input_ids"])
    with wired_limit(model):
        for uid, (name, prompt) in enumerate(prompts.items()):
            plain, plain_logits, plain_s = plain_greedy(model, prompt, args.tokens)
            spec, spec_logits, verifies, spec_s = speculative_greedy(model, drafter, prompt, args.tokens, uid)
            same = 0
            for a, b in zip(plain, spec):
                if a != b:
                    break
                same += 1
            print(f"  {name:<12} prompt={len(prompt)} tokens identical for {same}/{args.tokens}", flush=True)
            n = same
            if n:
                print(f"  {name:<12} logits, speculative vs plain: {summary(*score(plain_logits[:n], spec_logits[:n]))}", flush=True)
            print(
                f"  {name:<12} plain {args.tokens / plain_s:.2f} tok/s; drafted {args.tokens / spec_s:.2f} tok/s; "
                f"{args.tokens / verifies:.2f} tokens per verify",
                flush=True,
            )
            if same < args.tokens:
                print(f"  first difference at {same}: plain {tok.decode(plain[max(0, same - 3) : same + 3])!r} drafted {tok.decode(spec[max(0, same - 3) : same + 3])!r}", flush=True)
            mx.clear_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true", help="greedy: also use chat prompts, the model's own writing")
    parser.add_argument("mode", choices=["forced", "greedy", "accept"])
    parser.add_argument("checkpoint")
    parser.add_argument("drafter")
    parser.add_argument("reference")
    parser.add_argument("--variant", default="q8", choices=["bf16", "f16", "q8", "q6", "q4"])
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--embed-scale", default="target", choices=["target", "one"])
    parser.add_argument("--anchor", default="block", choices=["block", "query"])
    parser.add_argument("--no-nope", action="store_true", help="for comparison: rotate positions on every layer")
    parser.add_argument("--one-window", action="store_true", help="for comparison: one window for every layer")
    parser.add_argument("--segments", nargs="*")
    parser.add_argument("--limit", type=int, help="tokens to use from each text")
    parser.add_argument("--prefill", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=200)
    args = parser.parse_args()
    {"forced": cmd_forced, "greedy": cmd_greedy, "accept": cmd_accept}[args.mode](args)


if __name__ == "__main__":
    main()
