#!/usr/bin/env bash
# One-command, CPU-only run of the bundled prediction-only archive.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONHASHSEED=20261101
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
# Installation may access the package index; archive verification/training do not.
# Check exact direct pins so an old virtualenv cannot silently change the fit.
if ! .venv/bin/python - <<'PY' >/dev/null 2>&1
from importlib.metadata import version
from pathlib import Path
for line in Path('requirements.txt').read_text().splitlines():
    if line and not line.startswith('#'):
        name, expected = line.split('==')
        assert version(name) == expected, f'{name}: expected {expected}'
assert version('safemask-knee')
PY
then
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt -e .
fi
.venv/bin/python scripts/bootstrap_release.py --archive archive.zip --out data/release
if [[ -e data/release/expert_results.csv ]]; then
  echo "Official expert_results.csv detected. Run the full pipeline instead:" >&2
  echo "  .venv/bin/python -m safemask_knee all --release data/release --config configs/default.yaml" >&2
  exit 2
fi
mkdir -p outputs
.venv/bin/python -m safemask_knee train --release data/release --config configs/default.yaml | tee outputs/training_summary.json
echo "Training finished: reports/summary.md; outputs/debiased_training_weights.csv" >&2
