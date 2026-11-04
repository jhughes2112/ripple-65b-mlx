#!/usr/bin/env bash
# Downloads the pinned Ripple-65B checkpoint if needed, then writes a converted copy.
#
# usage: ./convert_weights.sh <f16|q8|q6|q4> [output dir]
#
# Set CHECKPOINT to use a copy you already have.
set -euo pipefail
cd "$(dirname "$0")"

REPO=ricursiveAI/Ripple-65B-Preview
REVISION=240d5966cbe00f0627c699e59ef58560b60303cf
CHECKPOINT=${CHECKPOINT:-models/Ripple-65B-Preview}
VARIANT=${1:?usage: ./convert_weights.sh <f16|q8|q6|q4> [output dir]}
OUT=${2:-models/Ripple-65B-Preview-$VARIANT}

if [ ! -x .venv-serve/bin/python ]; then
  echo "Run ./setup.sh first." >&2
  exit 1
fi

shards=$(find -L "$CHECKPOINT" -maxdepth 1 -name 'model-*-of-00016.safetensors' 2>/dev/null | wc -l | tr -d ' ')
if [ "$shards" != 16 ]; then
  echo "Downloading $REPO to $CHECKPOINT (about 130 GB)"
  .venv-serve/bin/hf download "$REPO" --revision "$REVISION" --local-dir "$CHECKPOINT"
fi

.venv-serve/bin/python convert.py "$CHECKPOINT" "$OUT" "$VARIANT" < /dev/null
