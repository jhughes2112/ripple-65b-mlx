# Ripple-65B on MLX

Runs [Ripple-65B Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) on Apple Silicon. The
upstream server needs NVIDIA GPUs and CUDA, so this unofficial port adds the model to
[mlx-lm](https://github.com/ml-explore/mlx-lm). It can also serve the model through a patched
[oMLX](https://github.com/jundot/omlx) with Ripple's own DFlash2 drafter, which is about twice as fast.

The repo holds only code. The weights come from Hugging Face.

## Which way to run it

| | Decode speed, 8-bit | Use it when |
|---|---:|---|
| **oMLX with the DFlash2 drafter** | 13–23 tok/s | You want speed. Also has batching and prompt caching |
| mlx-lm's server, `serve.py` | About 8.5 tok/s | You want the simplest setup, or replies that don't depend on the drafter |

## Prerequisites

### Hardware

- A Mac with Apple Silicon. Everything here was tested on one machine: an M5 Ultra with 256 GB, on macOS 27.2.
- Enough memory for the variant you choose, plus KV cache (about 42 GB at 128K context):

  | Variant | Weights | Recommended use |
  |---|---:|---|
  | `q8` | 68.8 GB | The best balance of speed and accuracy |
  | `f16` | 129.6 GB | The closest match to the reference |
  | `q6` | 52.6 GB | When `q8` doesn't fit |
  | BF16 | 129.6 GB | The checkpoint as shipped, but `f16` scores better |
  | `q4` | 36.4 GB | Not recommended: it changes the top token about 1 time in 10 |

- About 130 GB of disk for the checkpoint, plus any converted copy. The drafter adds 8 GB.

### Install these yourself

| Software | Why | How to get it |
|---|---|---|
| [git](https://git-scm.com/) | Clone this repo and oMLX | `xcode-select --install` |
| [uv](https://docs.astral.sh/uv/) | Creates the Python environments | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Full Xcode | oMLX only: builds its Metal kernels | App Store |

macOS already includes bash and curl, which the scripts and examples use.

### Installed for you

`setup.sh` installs Python 3.12, MLX 0.32.4, mlx-lm 0.32.2, transformers 5.22.1, tokenizers 0.23.3,
huggingface-hub 2.4.0 and numpy 2.5.4. Use these versions, because the results below were measured with
them or with the versions oMLX pins. mlx-lm 0.29.1 crashes in the server with MLX 0.32.4.

`setup_omlx.sh` clones oMLX at a pinned commit, applies `omlx/ripple-support.patch` and installs its own
environment. `./setup.sh --reference` adds a second environment with torch 2.15.0 and transformers
4.57.6, which only the accuracy checks need.

### The weights

The weights aren't in this repo, and no Hugging Face account is needed.

- [ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) at revision
  `240d5966cbe00f0627c699e59ef58560b60303cf` (129.6 GB). `convert_weights.sh` downloads it.
- [ricursiveAI/Ripple-65B-Preview-DFlash2](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2)
  at revision `35609b3be3dd08a822b3ba1257c1ea5dd6f87632` (8 GB). `setup_omlx.sh` downloads it.

You don't need a GGUF file, llama.cpp, or an NVIDIA GPU.

## Install and run with oMLX

```sh
git clone git@github.com:jhughes2112/ripple-65b-mlx.git
cd ripple-65b-mlx
./setup.sh
./convert_weights.sh q8 models/omlx/ripple-65b-q8
./setup_omlx.sh
```

`setup_omlx.sh` prints the command that starts the server. The model's id is `ripple-65b-q8`.

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "ripple-65b-q8", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
```

## Install and run with mlx-lm

```sh
./setup.sh
./convert_weights.sh q8
.venv-serve/bin/python serve.py --model models/Ripple-65B-Preview-q8 --host 127.0.0.1 --port 8080
```

Send requests with `"model": "default_model"`. Conversion takes seconds, so `serve.py` can also convert
the checkpoint as it loads, with no second copy on disk: `--bits 8` or `--f16` with
`--model models/Ripple-65B-Preview`. To use a checkpoint you already have, set `CHECKPOINT=/path/to/it`
when you run `convert_weights.sh`.

## Accuracy

The reference is the upstream PyTorch code using float32 math on the unmodified BF16 weights. Scores cover
10,768 positions across three long documents, up to 4,000 tokens each, over the full vocabulary, through
the cache in 512-token chunks.

| Variant | Same top token | Mean KL | p99 KL | Max KL |
|---|---:|---:|---:|---:|
| `f16` | 99.56% | 0.00020 | 0.0021 | 0.160 |
| BF16 | 99.27% | 0.00054 | 0.0057 | 0.219 |
| `q8` | 98.91% | 0.00098 | 0.0085 | 0.992 |

BF16 and `q8` compute the output head in float32. BF16 logits are coarse enough (steps of about 0.0625)
that exact ties for the top token are common, and this moved `q8` from 98.49% and BF16 from 98.67%. On an
earlier 5,000-position set, before that change, upstream's own BF16 code scored 99.14%, MLX BF16 98.94%,
6-bit 97.70% and 4-bit 89.38%.

**With the drafter, on the same positions, using real drafts and rollbacks past the window:** speculative
logits scored 98.53–98.73% against 98.47–98.93% for plain decoding, with slightly lower mean KL on all
three documents. This was measured before the float32 head change.

**Served by oMLX, drafter off vs. on:** log-probabilities differ by 0.002–0.007 on average.

**Sampling, at temperature 1:** samples matched the exact next-token probabilities computed from the
model, over 21 prefixes (total chi-square 95.4 where about 91 is expected, p = 0.31).

### Known limits

- **Replies can differ with the drafter.** At temperature 0, tokens match until a near-tie for the top
  token, which can come within 100 tokens. Rounding depends on how many rows are computed together.
  Neither reply is more correct.
- **Drafter settings not confirmed.** The sliding-window anchoring and embedding scale follow the config
  and mlx-vlm's behavior. Acceptance barely changed either way, so testing couldn't confirm them.
- **Sampling coverage.** The temperature 0.7 test's overall result was invalid because of how it handled a
  top-p boundary, though prefix by prefix it showed no bias. The tests use short replies, so they would
  catch a moderate bias, not a very slight one.
- **Drafter dtype.** Cast to F16 the drafter never produces an accepted draft, for a reason not yet known,
  so it stays in BF16 for Ripple.
- **Speed.** Measured with other work running. Depth 15 was no faster than the 8 drafts used by default.
- **Tool calls and reasoning.** Under `serve.py`, these aren't split into separate fields.

## Checking the port yourself

The reference runs the checkpoint's own Python code (`trust_remote_code`), so read
`modeling_ricursive.py` first. It takes several minutes per long document.

```sh
./setup.sh --reference
CK=models/Ripple-65B-Preview

# TEXTS holds the files you name, as NAME.txt.
.venv-reference/bin/python quality.py reference $CK $TEXTS ref.npz --math f32 --segments one two --seg-tokens 4000
.venv-serve/bin/python quality.py mlx $CK ref.npz --variant q8
.venv-serve/bin/python quality.py mlx models/Ripple-65B-Preview-q8 ref.npz --variant saved
```

| File | Purpose |
|---|---|
| `ricursive.py` | The model, ported from `modeling_ricursive.py` at revision `240d5966` |
| `serve.py` | mlx-lm's server, with optional conversion at load |
| `convert.py`, `convert_weights.sh` | Write a converted copy to disk |
| `setup.sh`, `setup_omlx.sh` | Create the environments |
| `omlx/ripple-support.patch` | What oMLX needs for Ripple and its drafter |
| `quality.py` | Score variants against a float32 reference |
| `dflash_check.py` | Speculative decoding through real drafts and rollbacks, using oMLX's environment |
| `check_served.py` | Compare served replies with each other or with a direct model |
| `sampling_check.py`, `sampling_truth.py` | Compare sampling with the drafter off and on, and with exact probabilities |
| `test_cache.py` | Sliding window and rollback on a small random model |
| `parity_torch.py`, `parity_mlx.py`, `bench_variants.py` | Earlier quick comparisons |

## Troubleshooting

- **`There is no Stream(gpu, 0) in current thread`:** use the pinned versions.
- **Decoding under 1 tok/s in your own script:** use mlx-lm's `wired_limit`. It made decoding about 7x
  faster here. `generate` and the server already apply it, but a direct `generate_step` loop doesn't.
- **A warning about an incorrect regex in the tokenizer:** this appears to be a false alarm. The suggested
  fix only splits digit groups into single digits, and the upstream server loads the tokenizer as shipped.
- **A prompt asking to run custom code:** the tokenizer loader prints this because the checkpoint's config
  names custom code. Serving doesn't need that code. Answer no. Converted copies don't trigger the prompt.

## Credits

Ripple-65B and the code this port is based on are by Ricursive Intelligence, under Apache 2.0. The drafter
code in the patch belongs to oMLX, also Apache 2.0.
