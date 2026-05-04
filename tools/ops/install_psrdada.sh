#!/usr/bin/env bash
#
# Install psrdada-python into the dsa110-rt conda env.
#
# This script needs more than just a plain pip install because:
#
# - TRASAL/psrdada-python's setup.py imports Cython without declaring it in
#   pyproject.toml [build-system].requires; PEP 517 isolated builds can't find
#   Cython. Solution: skip build isolation so the env's cython/setuptools/
#   wheel/numpy are visible to the build.
#
# - conda's MKL activation hook references MKL_INTERFACE_LAYER without
#   [ -z ${VAR:-} ] guards, which trips bash set -u. This script uses set -e
#   only; MKL_INTERFACE_LAYER is also defaulted below before sourcing conda.
#
# - libpsrdada was built from source on h01 under /usr/local/{include,lib}.
#   The conda env's compiler/linker search paths don't include /usr/local by
#   default, so CPATH and LD_LIBRARY_PATH are extended before pip build.
#
# Env pins for etcd3/protobuf and Cython<3 live in envs/dsa110-rt.yml.
#
set -eo pipefail

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-dsa110-rt}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export CPATH="${CPATH:+$CPATH:}/usr/local/include"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib"

echo "CPATH=$CPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "Installing psrdada-python into $ENV_NAME (no build isolation)..."
pip install --no-build-isolation --no-deps \
  'psrdada @ git+https://github.com/TRASAL/psrdada-python.git'
python -c "import psrdada; print('psrdada OK:', psrdada.__file__)"
