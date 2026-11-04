# Ripple-65B on MLX

An unofficial [mlx-lm](https://github.com/ml-explore/mlx-lm) port of
[ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview),
for running the model on Apple Silicon. The upstream server needs NVIDIA CUDA.

**Status: verified against the upstream PyTorch code on the real weights.** It picked
the same top token at all 1,500 positions of a long prompt. Greedy decoding through the
cache gave identical tokens. See the limits under Results.

## Files

- `ricursive.py`: the model, ported from `modeling_ricursive.py` at revision `240d5966`.
- `parity_torch.py`: writes reference logits using the upstream PyTorch code.
- `parity_mlx.py`: compares the port against those logits, then times decoding.
- `bench_variants.py`: speed and accuracy of BF16, F16 and 4/6/8-bit variants.

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

Apple M5 Ultra, 256 GB. Accuracy is scored against upstream PyTorch in BF16 on the CPU.

| Variant | Weights | Decode | Prefill | Same top token, greedy | KL, greedy | KL, short prompt |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 129.6 GB | 4.7 tok/s | 520 tok/s | 100% | 0.0005 | 0.0068 |
| F16 | 129.6 GB | 4.7 tok/s | 380 tok/s | 95.7% | 0.0003 | 0.0005 |
| 8-bit | 68.8 GB | 9.0 tok/s | 362 tok/s | 100% | 0.0015 | 0.025 |
| 6-bit | 52.6 GB | 10.6 tok/s | 382 tok/s | 100% | 0.0051 | 0.061 |
| 4-bit | 36.4 GB | 14.5 tok/s | 422 tok/s | 91.3% | 0.030 | 0.123 |

Limits of these numbers:

- Other work was running on the machine, so the speeds may be low.
- The accuracy samples are small: 17 positions in the short prompt and 23 in greedy decoding.
- The long prompt repeats one sentence, so it mostly tests the sliding window and cache,
  not how quantization affects varied text.
- Decoding is about 7x slower without mlx-lm's `wired_limit`. `generate` and the server
  apply it; a direct `generate_step` loop does not.

The tokenizer warning about an incorrect regex appears to be a false alarm. The suggested
fix only splits digit groups into single digits, and the upstream server loads the
checkpoint's tokenizer as shipped.
