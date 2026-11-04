# Ripple-65B on MLX

Runs [Ripple-65B Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) on Apple Silicon.
The upstream server needs NVIDIA GPUs and CUDA, so this unofficial port adds the model to
[mlx-lm](https://github.com/ml-explore/mlx-lm). It also includes tools to measure how closely the port
matches the upstream code.

The repo holds only code. The weights come from Hugging Face.

## Prerequisites

### Hardware

- A Mac with Apple Silicon. Everything here was tested on one machine: an M5 Ultra with 256 GB, on macOS 27.2.
- Enough memory for the variant you choose, plus about 4 GB:

  | Variant | Weights | Recommended use |
  |---|---:|---|
  | `q8` | 68.8 GB | The best balance of speed and accuracy |
  | `f16` | 129.6 GB | The closest match to the reference |
  | `q6` | 52.6 GB | When `q8` doesn't fit |
  | BF16 | 129.6 GB | The checkpoint as shipped, but `f16` scores better |
  | `q4` | 36.4 GB | Not recommended: it changes the top token about 1 time in 10 |

- About 130 GB of disk for the checkpoint, plus the size of any converted copy.

### Install these yourself

| Software | Why | How to get it |
|---|---|---|
| [git](https://git-scm.com/) | Clone this repo | `xcode-select --install` |
| [uv](https://docs.astral.sh/uv/) | Creates the Python environments | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

macOS already includes bash and curl, which the scripts and examples use.

### Installed for you by `setup.sh`

| Package | Version | Why |
|---|---:|---|
| Python | 3.12 | Runs everything |
| [MLX](https://github.com/ml-explore/mlx) | 0.32.4 | Runs the model on the GPU |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | 0.32.2 | Loads the model and provides the server |
| transformers | 5.22.1 | Tokenizer |
| tokenizers | 0.23.3 | Tokenizer |
| huggingface-hub | 2.4.0 | Downloads the weights |
| numpy | 2.5.4 | Scoring |

Use these versions, because the results below were measured with them. mlx-lm 0.29.1 crashes in the
server with MLX 0.32.4. To check the port against the upstream code, `./setup.sh --reference` also
creates a second environment with torch 2.15.0 and transformers 4.57.6.

### The weights

The weights aren't in this repo. You need
[ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) at revision
`240d5966cbe00f0627c699e59ef58560b60303cf` (129.6 GB, Apache 2.0).

`convert_weights.sh` downloads them for you. No Hugging Face account is needed. To download them
yourself after running `setup.sh`:

```sh
.venv-serve/bin/hf download ricursiveAI/Ripple-65B-Preview \
  --revision 240d5966cbe00f0627c699e59ef58560b60303cf --local-dir models/Ripple-65B-Preview
```

You don't need a GGUF file, llama.cpp, or an NVIDIA GPU.

## Install

```sh
git clone git@github.com:jhughes2112/ripple-65b-mlx.git
cd ripple-65b-mlx
./setup.sh
./convert_weights.sh q8
```

That creates `.venv-serve`, downloads the checkpoint to `models/Ripple-65B-Preview` if it isn't there,
and writes the 8-bit copy to `models/Ripple-65B-Preview-q8`.

## Run

Start an OpenAI-compatible server on the copy:

```sh
.venv-serve/bin/python serve.py --model models/Ripple-65B-Preview-q8 --host 127.0.0.1 --port 8080
```

Then send it a request:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "default_model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
```

### Without a second copy

Conversion takes seconds, so `serve.py` can also convert the checkpoint as it loads. This saves disk
and costs a little time at each start:

```sh
.venv-serve/bin/python serve.py --bits 8 --model models/Ripple-65B-Preview --host 127.0.0.1 --port 8080
.venv-serve/bin/python serve.py --f16 --model models/Ripple-65B-Preview --host 127.0.0.1 --port 8080
```

`convert_weights.sh` downloads a pinned revision of the checkpoint. To use a copy you already have,
set `CHECKPOINT=/path/to/checkpoint`.

## Accuracy

The reference is the upstream PyTorch code using float32 math on the unmodified BF16 weights. Scores
cover 5,000 positions across five texts (code, docs, English, multilingual, an agent transcript), over
the full vocabulary, through the rotating cache in 512-token chunks.

| Variant | Same top token | Mean KL | p99 KL | Max KL | Decode |
|---|---:|---:|---:|---:|---:|
| `f16` | 99.66% | 0.00013 | 0.0013 | 0.021 | 4.7 tok/s |
| Upstream PyTorch, BF16 (yardstick) | 99.14% | 0.00041 | 0.0031 | 0.164 | CPU only |
| BF16 | 98.94% | 0.00056 | 0.0050 | 0.099 | 4.7 tok/s |
| `q8` | 98.66% | 0.00079 | 0.0075 | 0.122 | 9.0 tok/s |
| `q6` | 97.70% | 0.0034 | 0.029 | 0.97 | 10.6 tok/s |
| `q4` | 89.38% | 0.081 | 0.94 | 7.7 | 14.5 tok/s |

MLX BF16 sits close to the upstream code's own BF16 result, so the port adds little error of its own.
`f16` is closer to float32 than upstream BF16 is.

**Served replies against the same variant loaded directly**, with 64-token greedy replies to four prompts:

| Variant | Same top token | Largest log-probability difference |
|---|---:|---:|
| `q8` | 163 of 164 | 0.104 |
| `f16` | 170 of 171 | 0.008 |

The `q8` miss is a near-tie (log-probabilities -1.06 and -1.04).

### Known limits

- The cause of the 0.104 `q8` gap is not established. Splitting the prompt the way the server does
  didn't remove it.
- Speeds were measured with other work running on the machine, so they may be low.
- These scores are teacher-forced. They don't show how far long free-running replies drift.
- `q8` and the other variants use plain, uncalibrated quantization (affine, groups of 64).
- The model card gives no sampling settings.
- Tool calls and reasoning aren't split into separate fields. The upstream server has parsers for
  that, and `serve.py` returns the model's raw text.

## Checking the port yourself

The reference runs the checkpoint's own Python code (`trust_remote_code`), so read
`modeling_ricursive.py` first. It takes about 5 to 30 minutes on an M5 Ultra.

```sh
./setup.sh --reference
CK=models/Ripple-65B-Preview

# TEXTS holds code.txt, docs.txt, english.txt, multilingual.txt and agent.txt.
.venv-reference/bin/python quality.py reference $CK $TEXTS ref-f32.npz --math f32
.venv-serve/bin/python quality.py mlx $CK ref-f32.npz --variant q8

# A converted copy, scored as saved:
.venv-serve/bin/python quality.py mlx models/Ripple-65B-Preview-q8 ref-f32.npz --variant saved
```

The texts aren't in this repo. The results above used the set in Strata's `data/quant_quality/v1`, but
any five files with those names will work.

| File | Purpose |
|---|---|
| `ricursive.py` | The model, ported from `modeling_ricursive.py` at revision `240d5966` |
| `serve.py` | mlx-lm's server, with optional conversion at load |
| `convert.py`, `convert_weights.sh` | Write a converted copy to disk |
| `setup.sh` | Create the environments and link the model into mlx-lm |
| `quality.py` | Score variants against a float32 reference |
| `check_served.py` | Compare served replies with the same variant loaded directly |
| `parity_torch.py`, `parity_mlx.py` | Quick comparison, including greedy decoding through the cache |
| `bench_variants.py` | Speed and accuracy of one variant against `parity_torch.py`'s output |

## Troubleshooting

- **`There is no Stream(gpu, 0) in current thread`:** this appears with mlx-lm 0.29.1 and MLX 0.32.4.
  Use the pinned versions.
- **Decoding under 1 tok/s in your own script:** use mlx-lm's `wired_limit`. It made decoding about 7x
  faster here. `generate` and the server already apply it, but a direct `generate_step` loop doesn't.
- **A warning about an incorrect regex in the tokenizer:** this appears to be a false alarm. The
  suggested fix only splits digit groups into single digits, and the upstream server loads the
  tokenizer as shipped.
- **A prompt asking to run custom code:** the tokenizer loader prints this because the checkpoint's
  config names custom code. Serving doesn't need that code, and `.venv-serve` has no torch to run it.
  Answer no. Converted copies don't trigger the prompt.

## Credits

Ripple-65B and the code this port is based on are by Ricursive Intelligence, under Apache 2.0.
