#!/usr/bin/env bash
# Install the dsart package (this repo) into the dsa110-rt conda env in
# editable mode. Idempotent — safe to re-run; pip will skip if already
# installed unless --force-reinstall is passed.
#
# Part of the canonical 3-step env setup per plan §6:
#   (1) conda env create -f envs/dsa110-rt.yml
#   (2) tools/ops/install_psrdada.sh
#   (3) tools/ops/install_dsart.sh    <-- this script
set -eo pipefail   # -u DROPPED: conda's MKL hook references MKL_INTERFACE_LAYER
                   # without a guard. Same issue + fix as install_psrdada.sh.
                   # See plan §6 (Conda-activate shell pattern).

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-dsa110-rt}"
REPO_ROOT="${REPO_ROOT:-$HOME/proj/dsa110-rt}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-GNU,LP64}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"
echo "Installing dsart (editable, --no-deps) into $ENV_NAME ..."
pip install -e . --no-deps
python -c "import dsart; print('dsart OK:', dsart.__file__)"
python -c "import dsart.common.config_loader; print('config_loader OK')"
python -c "import dsart.common.host; print('host OK')"
