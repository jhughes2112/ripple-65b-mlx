# Ripple-65B on MLX

An unofficial [mlx-lm](https://github.com/ml-explore/mlx-lm) port of
[ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview),
for running the model on Apple Silicon. The upstream server needs NVIDIA CUDA.

**Status: checked against the upstream PyTorch code on the real weights.** See Results for the
scores and their limits.

## Files

- `ricursive.py`: the model, ported from `modeling_ricursive.py` at revision `240d5966`.
- `parity_torch.py`: writes reference logits using the upstream PyTorch code.
- `parity_mlx.py`: compares the port against those logits, then times decoding.
- `bench_variants.py`: speed and accuracy of BF16, F16 and 4/6/8-bit variants.
- `quality.py`: position-by-position scoring on varied text against a float32 reference.
- `serve.py`: mlx-lm's server, converting to F16 or quantizing in memory at load.
- `check_served.py`: compares served replies with the same variant loaded directly.

## Install

Use a dedicated venv, with `transformers==4.57.6` and `mlx-lm`. Then link the model
into mlx-lm, which finds models by their `model_type`:

```sh
ln -s "$PWD/ricursive.py" "$(python -c 'import mlx_lm,os;print(os.path.dirname(mlx_lm.__file__))')/models/ricursive.py"
```

## Check the port

```sh
python parity_torch.py /path/to/checkpoint reference.npz
python parity_mlx.py /path/to/checkpoint reference.npz
```

`parity_torch.py` runs Python code from the model repo, so review it first.

## Results

Apple M5 Ultra, 256 GB, MLX 0.32.4, mlx-lm 0.32.2. The reference is the upstream PyTorch code using
float32 math on the unmodified BF16 weights (`quality.py`). Scores cover 5,000 positions across five
texts (code, docs, English, multilingual, an agent transcript), over the full vocabulary, through the
rotating cache in 512-token chunks.

| Variant | Weights | Same top token | Mean KL | p99 KL | Max KL | Decode |
|---|---:|---:|---:|---:|---:|---:|
| F16 | 129.6 GB | 99.66% | 0.00013 | 0.0013 | 0.021 | 4.7 tok/s |
| Upstream PyTorch, BF16 (yardstick) | 129.6 GB | 99.14% | 0.00041 | 0.0031 | 0.164 | CPU |
| BF16 | 129.6 GB | 98.94% | 0.00056 | 0.0050 | 0.099 | 4.7 tok/s |
| 8-bit | 68.8 GB | 98.66% | 0.00079 | 0.0075 | 0.122 | 9.0 tok/s |
| 6-bit | 52.6 GB | 97.70% | 0.0034 | 0.029 | 0.97 | 10.6 tok/s |
| 4-bit | 36.4 GB | 89.38% | 0.081 | 0.94 | 7.7 | 14.5 tok/s |

MLX BF16 sits close to the upstream code's own BF16 result, so the port adds little error of its own.
F16 is closer to float32 than upstream BF16 is. 4-bit loses too much to recommend.

**Serving path** (`check_served.py`): 64-token greedy replies to four prompts through `serve.py`,
compared with the same variant loaded directly.

| Variant | Same top token | Largest log-probability difference |
|---|---:|---:|
| 8-bit | 163 of 164 | 0.104 |
| F16 | 170 of 171 | 0.008 |

The 8-bit miss is a near-tie (log-probabilities -1.06 and -1.04). The cause of the 8-bit gap is not
established. Feeding the prompt in the same split as the server did not remove it.

Limits of these numbers:

- Speeds were measured with other work running, so they may be low.
- Teacher-forced scores don't show how much long free-running replies would diverge.
- Gateway or other proxy layers are not covered by these tests.
- `serve.py` needs a current mlx-lm. mlx-lm 0.29.1 with MLX 0.32.4 crashes in the server thread
  (`There is no Stream(gpu, 0) in current thread`).
- `parity_torch.py` and `quality.py reference` need `transformers==4.57.6` and torch, so they use a
  separate venv from serving.
- Decoding is about 7x slower without mlx-lm's `wired_limit`. `generate` and the server apply it;
  a direct `generate_step` loop does not.
- The tokenizer warning about an incorrect regex appears to be a false alarm. The suggested fix only
  splits digit groups into single digits, and the upstream server loads the tokenizer as shipped.
