#!/usr/bin/env bash
# Creates the Python environments and links the model into mlx-lm.
#
# usage: ./setup.sh [--reference]
#
#   .venv-serve      running, converting and scoring on MLX
#   .venv-reference  the upstream PyTorch code (--reference only)
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv venv --allow-existing --python 3.12 .venv-serve
uv pip install --python .venv-serve/bin/python -r requirements-serve.txt

# mlx-lm finds models by model_type, in its models folder.
models=$(.venv-serve/bin/python -c 'import mlx_lm, os; print(os.path.join(os.path.dirname(mlx_lm.__file__), "models"))')
ln -sf "$PWD/ricursive.py" "$models/ricursive.py"
.venv-serve/bin/python -c 'from mlx_lm.models import ricursive'

if [ "${1:-}" = "--reference" ]; then
  uv venv --allow-existing --python 3.12 .venv-reference
  uv pip install --python .venv-reference/bin/python -r requirements-reference.txt
fi

echo "Setup complete."
