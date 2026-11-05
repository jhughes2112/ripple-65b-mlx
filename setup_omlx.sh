#!/usr/bin/env bash
# Builds a patched oMLX for Ripple-65B and downloads the DFlash2 drafter.
#
# usage: ./setup_omlx.sh [models folder holding ripple-65b-q8]
#
# Run ./setup.sh and ./convert_weights.sh first. Needs full Xcode for oMLX's Metal kernels.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD

OMLX_URL=https://github.com/jundot/omlx.git
OMLX_COMMIT=4213c328
DRAFTER_REPO=ricursiveAI/Ripple-65B-Preview-DFlash2
DRAFTER_REVISION=35609b3be3dd08a822b3ba1257c1ea5dd6f87632
MODELS=${1:-models/omlx}
MODELS=$(cd "$MODELS" && pwd)
DRAFTER=$HERE/models/Ripple-65B-Preview-DFlash2
STATE=$HERE/models/omlx-state
OMLX=$HERE/omlx-build

if [ ! -x .venv-serve/bin/hf ]; then
  echo "Run ./setup.sh first." >&2
  exit 1
fi
if [ ! -f "$DRAFTER/model.safetensors" ]; then
  echo "Downloading the drafter (about 8 GB)"
  .venv-serve/bin/hf download "$DRAFTER_REPO" --revision "$DRAFTER_REVISION" --local-dir "$DRAFTER"
fi

if [ ! -d "$OMLX" ]; then
  git clone "$OMLX_URL" "$OMLX"
  git -C "$OMLX" checkout -q "$OMLX_COMMIT"
  git -C "$OMLX" apply "$HERE/omlx/ripple-support.patch"
fi
uv venv --allow-existing --seed --python 3.12 "$OMLX/.venv"
make -C "$OMLX" install PYTHON=.venv/bin/python

# mlx-lm finds models by model_type, in its models folder.
models=$("$OMLX/.venv/bin/python" -c 'import mlx_lm, os; print(os.path.join(os.path.dirname(mlx_lm.__file__), "models"))')
ln -sf "$HERE/ricursive.py" "$models/ricursive.py"
"$OMLX/.venv/bin/python" -c 'from mlx_lm.models import ricursive'

mkdir -p "$STATE"
cat > "$STATE/model_settings.json" <<JSON
{"version": 1, "models": {"ripple-65b-q8": {"dflash_enabled": true, "dflash_draft_model": "$DRAFTER", "dflash_draft_window_size": 16384, "dflash_block_size": 9, "dflash_verify_mode": "adaptive"}}}
JSON

echo
echo "Setup complete. Start the server with:"
echo "  cd $OMLX && HF_HUB_OFFLINE=1 .venv/bin/omlx serve --model-dir $MODELS --base-path $STATE --host 127.0.0.1 --port 8080 --memory-guard off --headless"
