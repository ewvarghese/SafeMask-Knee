#!/usr/bin/env bash
# One-command, CPU-only run of the bundled prediction-only archive.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONHASHSEED=20261101
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# The pinned scientific wheels are for Python 3.10–3.13, not the user's WSL
# Python 3.14. Prefer a compatible interpreter even when `python3` is newer.
python_bin="${SAFEMASK_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  for candidate in python3.11 python3.12 python3.13 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] < (3, 14))' >/dev/null 2>&1; then
      python_bin="$candidate"
      break
    fi
  done
fi
if [[ -z "$python_bin" ]] || ! "$python_bin" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] < (3, 14))' >/dev/null 2>&1; then
  echo "No supported Python found in this Linux/WSL shell (need 3.10–3.13; use 3.11)." >&2
  echo "Python 3.14 is incompatible with the pinned scientific dependencies." >&2
  echo "Use native Windows PowerShell with 'py -3.11' as shown in README.md," >&2
  echo "or install Python 3.11 plus python3.11-venv inside WSL." >&2
  exit 2
fi
version="$("$python_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

# A failed WSL `venv` may contain bin/python but no pip. Repair it rather
# than treating that half-created environment as valid. Don't mix versions.
existing="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
if [[ ! -x .venv/bin/python ]] || [[ "$existing" != "$version" ]] || \
   ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  if ! "$python_bin" -m venv --clear .venv; then
    echo "Could not create .venv: Python $version needs ensurepip in this Linux/WSL shell." >&2
    echo "On Debian/Ubuntu: sudo apt update && sudo apt install python${version}-venv" >&2
    echo "Then rerun: bash scripts/run_release.sh" >&2
    echo "Or use native Windows PowerShell with 'py -3.11' (without bash)." >&2
    exit 2
  fi
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
.venv/bin/python -m safemask_knee doctor --release data/release --archive archive.zip \
  --config configs/default.yaml | tee outputs/doctor_cli_summary.json
.venv/bin/python -m safemask_knee train --release data/release --config configs/default.yaml | tee outputs/training_summary.json
echo "Training finished: reports/summary.md; outputs/debiased_training_weights.csv" >&2
