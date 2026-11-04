"""Does the serving path keep the logits? Compares a served model to the same variant loaded directly.

usage:
  check_served.py capture <base url> <model id> <out.json>
      Sends fixed prompts at temperature 0 through the gateway and saves tokens and logprobs.
  check_served.py compare <checkpoint> <captured.json> --variant bf16|f16|q8|q6|q4
      Replays the served tokens through the directly loaded model. Each served token should be
      the direct model's top token, with the same log-probability.
"""

import argparse
import json
import urllib.request

PROMPTS = [
    "Explain what a mutex is in two sentences.",
    "Write a Python function that returns the n-th Fibonacci number, with a short docstring.",
    "Traduis en français : The lighthouse at the end of the breakwater had been automated for years.",
    "A train leaves at 3:15 pm and travels 180 km at 72 km/h. When does it arrive? Show your steps.",
]
MAX_TOKENS = 64


def capture(args):
    results = []
    for prompt in PROMPTS:
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS,
            "logprobs": True,
            "top_logprobs": 3,
            "stream": False,
        }
        request = urllib.request.Request(
            args.url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            reply = json.load(response)
        choice = reply["choices"][0]
        results.append(
            {
                "prompt": prompt,
                "content": choice["message"].get("content"),
                "finish": choice.get("finish_reason"),
                "usage": reply.get("usage"),
                "logprobs": choice.get("logprobs"),
            }
        )
        print(repr(choice["message"].get("content"))[:150], choice.get("finish_reason"), reply.get("usage"), flush=True)
    json.dump(results, open(args.out, "w"))
    print("logprobs keys:", list((results[0]["logprobs"] or {}).keys()), flush=True)


def compare(args):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_lm import load
    from mlx_lm.generate import wired_limit

    results = json.load(open(args.captured))
    model, tok = load(args.checkpoint, lazy=True)
    if args.variant == "f16":
        model.set_dtype(mx.float16)
    elif args.variant.startswith("q"):
        nn.quantize(model, group_size=64, bits=int(args.variant[1:]))
    mx.eval(model.parameters())

    total = same_top = 0
    worst = 0.0
    with wired_limit(model):
        for item in results:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}], add_generation_prompt=True
            )
            if not isinstance(prompt, list):
                prompt = list(prompt["input_ids"])
            served = [int(entry["id"]) for entry in item["logprobs"]["content"]]
            logps = [entry["logprob"] for entry in item["logprobs"]["content"]]
            usage = item["usage"] or {}
            if usage.get("prompt_tokens") != len(prompt):
                print(f"PROMPT LENGTH DIFFERS: served {usage.get('prompt_tokens')} direct {len(prompt)}", flush=True)
            cache = model.make_cache()
            logits = model(mx.array(prompt)[None], cache=cache)[0, -1]
            matches = 0
            for token, logp in zip(served, logps):
                row = np.array(logits.astype(mx.float32), dtype=np.float64)
                row = row - row.max()
                direct = row - np.log(np.exp(row).sum())
                matches += int(direct.argmax()) == token
                worst = max(worst, abs(float(direct[token]) - float(logp)))
                logits = model(mx.array([token])[None], cache=cache)[0, -1]
            total += len(served)
            same_top += matches
            print(f"{matches}/{len(served)} served tokens are the direct top token; {item['finish']}", flush=True)
    print(
        f"[{args.variant}] served vs direct: {same_top}/{total} same top token; "
        f"largest log-probability difference {worst:.6f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("capture")
    p.add_argument("url")
    p.add_argument("model")
    p.add_argument("out")
    p.set_defaults(run=capture)
    p = commands.add_parser("compare")
    p.add_argument("checkpoint")
    p.add_argument("captured")
    p.add_argument("--variant", required=True)
    p.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
