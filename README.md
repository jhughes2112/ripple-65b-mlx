# Ripple-65B on MLX

Runs [Ripple-65B Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) on Apple Silicon. The upstream server
needs NVIDIA GPUs and CUDA, so this unofficial port adds the model to [mlx-lm](https://github.com/ml-explore/mlx-lm) and
to a patched [oMLX](https://github.com/jundot/omlx).

**Status: finished as is, and not being developed further.** It works, but the model is slow on a Mac, with about 19 tok/s
sustained on an M5 Ultra using 8-bit weights and the drafter. Don't expect fixes or updates.

The repo holds only code. The weights come from Hugging Face, and no account is needed.

## What you need

- A Mac with Apple Silicon. Everything here was tested on one machine: an M5 Ultra with 256 GB, on macOS 27.2.
- Memory for the weights plus KV cache, which is about 42 GB at 128K context. The 8-bit server with the drafter peaked at
  187 GB here.
- About 130 GB of disk for the checkpoint, plus the converted copy. The drafter adds 8 GB.
- [git](https://git-scm.com/) (`xcode-select --install`) and [uv](https://docs.astral.sh/uv/)
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Full Xcode from the App Store, for oMLX only. It builds oMLX's Metal kernels.

The scripts install pinned versions of everything else. Keep those versions, because the results below depend on them.

## Which weights to use

| Variant | Size | Same top token | Mean KL | Use it when |
|---|---:|---:|---:|---|
| `q8` | 68.8 GB | 98.91% | 0.00098 | **Recommended:** the best balance of speed and accuracy |
| `f16` | 129.6 GB | 99.56% | 0.00020 | You want the closest match to the reference |
| BF16 | 129.6 GB | 99.27% | 0.00054 | You want the checkpoint exactly as shipped, though `f16` scores better |
| oQ5e | About 46 GB | 97.20% | 0.00481 | `q8` doesn't fit. It's mixed 5, 6 and 8-bit, averaging 5.70 bits per weight |
| `q6` | 52.6 GB | 97.70% | Not measured | Not needed: oQ5e is smaller |
| `q4` | 36.4 GB | 89.38% | Not measured | Not recommended: it changes the top token about 1 time in 10 |

The `q6` and `q4` scores come from an earlier 5,000-position test, so they're only roughly comparable. See
[Accuracy](#accuracy) for how the rest were measured.

## Run it with oMLX and the drafter

This is the faster way, at 13–23 tok/s with `q8`, and it adds batching and prompt caching.

```sh
git clone git@github.com:jhughes2112/ripple-65b-mlx.git
cd ripple-65b-mlx
./setup.sh
./convert_weights.sh q8 models/omlx/ripple-65b-q8
./setup_omlx.sh
```

`convert_weights.sh` downloads the 129.6 GB checkpoint unless you already have it, in which case set
`CHECKPOINT=/path/to/it`. `setup_omlx.sh` downloads the drafter, then prints the command that starts the server. Run that
command, then send a request:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "ripple-65b-q8", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
```

## Run it with mlx-lm

This is the simplest way, at about 8.5 tok/s with `q8` and no drafter.

```sh
./setup.sh
./convert_weights.sh q8
.venv-serve/bin/python serve.py --model models/Ripple-65B-Preview-q8 --host 127.0.0.1 --port 8080
```

Send requests with `"model": "default_model"`. `serve.py` can also convert as it loads, with no second copy on disk: use
`--model models/Ripple-65B-Preview` with `--bits 8` or `--f16`.

## Convert to oQ5e

Do the oMLX setup above first, because the build needs the patched oMLX, the BF16 checkpoint and the `q8` copy. Use
absolute paths, as shown, because relative ones break the build's links to the `q8` copy.

```sh
R=$PWD
mkdir -p "$R/models/oq"
cd "$R/omlx-build"
.venv/bin/python "$R/build_oq.py" 5 \
  "$R/models/Ripple-65B-Preview" \
  "$R/models/omlx/ripple-65b-q8" \
  "$R/models/oq"
```

The result is `models/oq/ripple-65b-oq5e`. The arguments are the oQ level, the BF16 checkpoint, the `q8` copy and the output
folder. The first run also collects an imatrix and saves it in `models/oq/.oqe_imatrix`, so later builds skip that step.

- **Run it on an otherwise idle Mac.** Calibration loads the whole `q8` copy. Running a large job next to a loaded model
  exhausted memory and crashed the test machine.
- **Calibration uses `q8`, not BF16.** oQ's own fallback when a model is too big to calibrate in memory is a uniform 4-bit
  copy, and `q8` is much closer to the original. Calibrating on BF16 might score better, but that was never tried.
- **Only level 5 has been built and scored.** Other levels should work with the same command, with no promises on quality.

### Serve oQ5e

**This hasn't been tested.** oQ5e was scored by loading it directly, using oMLX's fast verify kernels, but never through the
server. To try it, replace the settings file with an entry for oQ5e, using the drafter settings tested with `q8`:

```sh
cat > "$R/models/omlx-state/model_settings.json" <<JSON
{"version": 1, "models": {"ripple-65b-oq5e": {"dflash_enabled": true, "dflash_draft_model": "$R/models/Ripple-65B-Preview-DFlash2", "dflash_draft_window_size": 16384, "dflash_block_size": 9, "dflash_verify_mode": "adaptive"}}}
JSON
cd "$R/omlx-build"
HF_HUB_OFFLINE=1 .venv/bin/omlx serve --model-dir "$R/models/oq" --base-path "$R/models/omlx-state" \
  --host 127.0.0.1 --port 8080 --memory-guard off --headless
```

The model's id is `ripple-65b-oq5e`.

## Accuracy

The reference is the upstream PyTorch code using float32 math on the unmodified BF16 weights. Scores cover 10,768 positions
across three long documents, over the full vocabulary, through the cache in 512-token chunks.

| Variant | Same top token | Mean KL | p99 KL |
|---|---:|---:|---:|
| `f16` | 99.56% | 0.00020 | 0.0021 |
| BF16 | 99.27% | 0.00054 | 0.0057 |
| `q8` | 98.91% | 0.00098 | 0.0085 |
| oQ5e | 97.20% | 0.00481 | 0.0439 |

**At long context:** a further 1,000 positions at each of two depths, using the fast verify kernels and 5-row verification:

| Context | Variant | Same top token | Mean KL | p99 KL |
|---:|---|---:|---:|---:|
| 60K | `q8` | 99.00% | 0.0090 | 0.081 |
| 60K | oQ5e | 98.00% | 0.0173 | 0.432 |
| 80K | `q8` | 99.60% | 0.00038 | 0.0064 |
| 80K | oQ5e | 99.00% | 0.00239 | 0.0304 |

**With the drafter:** `q8` speculative logits scored 98.53–98.73% against 98.47–98.93% for plain decoding, measured before
the output head moved to float32. Served logits with the drafter off and on differ by 0.002–0.007 on average. At temperature
1, samples matched the exact next-token probabilities over 21 prefixes (p = 0.31).

## Known limits

- **oQ5e isn't as close as `q8`.** Its mean KL is about five times higher on the standard test. A partial investigation
  suggests that oQ may not be spending its extra bits where they help most: restoring all the feed-forward layers to `q8`
  closed about 40% of the gap, but restoring `down_proj`, where oQ spent most of its feed-forward boosts, closed none of it.
  That investigation wasn't finished.
- **Replies can differ with the drafter.** At temperature 0, tokens match until a near-tie for the top token, which can come
  within 100 tokens. Neither reply is more correct.
- **Drafter settings aren't confirmed.** The sliding-window anchoring and embedding scale follow the config and mlx-vlm's
  behavior. Acceptance barely changed either way, so testing couldn't confirm them.
- **The drafter stays in BF16.** Cast to F16 it never produces an accepted draft, for a reason not yet known.
- **Speed figures aren't clean benchmarks.** They were measured with other work running.
- **Tool calls and reasoning.** Under `serve.py`, these aren't split into separate fields.

## Troubleshooting

- **`There is no Stream(gpu, 0) in current thread`:** use the pinned versions.
- **Decoding under 1 tok/s in your own script:** use mlx-lm's `wired_limit`. `generate` and the server already apply it, but a
  direct `generate_step` loop doesn't.
- **A warning about an incorrect regex in the tokenizer:** this appears to be a false alarm. The upstream server loads the
  tokenizer as shipped.
- **A prompt asking to run custom code:** answer no. Serving doesn't need that code, and converted copies don't trigger the
  prompt.

## Checking the port yourself

The reference runs the checkpoint's own Python code (`trust_remote_code`), so read `modeling_ricursive.py` first.

```sh
./setup.sh --reference
CK=models/Ripple-65B-Preview

# TEXTS is a folder you provide, holding the files you name as NAME.txt.
.venv-reference/bin/python quality.py reference $CK $TEXTS ref.npz --math f32 --segments one two --seg-tokens 4000
.venv-serve/bin/python quality.py mlx $CK ref.npz --variant q8
.venv-serve/bin/python quality.py mlx models/Ripple-65B-Preview-q8 ref.npz --variant saved
```

| File | Purpose |
|---|---|
| `ricursive.py` | The model, ported from `modeling_ricursive.py` at revision `240d5966` |
| `serve.py` | mlx-lm's server, with optional conversion at load |
| `convert.py`, `convert_weights.sh` | Write a converted copy to disk |
| `build_oq.py` | Build an oQe quant, calibrating on the `q8` copy |
| `setup.sh`, `setup_omlx.sh` | Create the environments |
| `omlx/ripple-support.patch` | What oMLX needs for Ripple, its drafter and oQ calibration |
| `quality.py` | Score variants against a float32 reference |
| `dflash_check.py` | Speculative decoding through real drafts and rollbacks, using oMLX's environment |
| `check_served.py` | Compare served replies with each other or with a direct model |
| `sampling_check.py`, `sampling_truth.py` | Compare sampling with the drafter off and on, and with exact probabilities |
| `test_cache.py` | Sliding window and rollback on a small random model |
| `parity_torch.py`, `parity_mlx.py`, `bench_variants.py` | Earlier quick comparisons |

Pinned sources: the checkpoint at revision `240d5966cbe00f0627c699e59ef58560b60303cf`, the
[drafter](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2) at `35609b3be3dd08a822b3ba1257c1ea5dd6f87632`, and
oMLX at `4213c328`.

## Credits

Ripple-65B and the code this port is based on are by Ricursive Intelligence, under Apache 2.0. The drafter code in the patch
belongs to oMLX, also Apache 2.0.
