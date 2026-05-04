#!/usr/bin/env bash
#
# Install psrdada-python into the dsa110-rt conda env.
# Split out from envs/dsa110-rt.yml because TRASAL/psrdada-python's
# setup.py imports Cython without declaring it in pyproject.toml's
# [build-system].requires; PEP 517 isolated builds can't find Cython.
# Solution: skip build isolation; the env's cython/setuptools/wheel/numpy
# are visible to the build directly.
#
set -euo pipefail
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-dsa110-rt}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "Installing psrdada-python into $ENV_NAME (no build isolation)..."
pip install --no-build-isolation --no-deps \
  'psrdada @ git+https://github.com/TRASAL/psrdada-python.git'
python -c "import psrdada; print('psrdada OK:', psrdada.__file__)"
