#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=20261101

if [[ $# -ne 1 ]]; then
  echo "Usage: scripts/run_all.sh <RELEASE_DIR>" >&2
  exit 2
fi

python -m pytest -q
python -m safemask_knee all --release "$1" --config configs/default.yaml
