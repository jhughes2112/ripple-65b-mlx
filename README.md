# Ripple-65B on MLX

An unofficial [mlx-lm](https://github.com/ml-explore/mlx-lm) port of
[ricursiveAI/Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview),
for running the model on Apple Silicon. The upstream server needs NVIDIA CUDA.

**Status: not yet verified on the real weights.** So far the port has only been
tested on small random-weight models.

## Files

- `ricursive.py`: the model, ported from `modeling_ricursive.py` at revision `240d5966`.
- `parity_torch.py`: writes reference logits using the upstream PyTorch code.
- `parity_mlx.py`: compares the port against those logits, then times decoding.

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
