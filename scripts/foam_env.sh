#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/python_env.sh"
if [[ "${WM_PROJECT_VERSION:-}" != '13' ]]; then
  foamrc="$(python -c 'import config; print(config.FOAM13_BASHRC)')"
  if [[ ! -f "$foamrc" && -f "$HOME/OpenFOAM/OpenFOAM-13/etc/bashrc" ]]; then
    foamrc="$HOME/OpenFOAM/OpenFOAM-13/etc/bashrc"
  fi
  [[ -f "$foamrc" ]] || { echo "[env] Foundation OpenFOAM 13 not found at $foamrc" >&2; exit 2; }
  set +u
  source "$foamrc"
  set -u
fi
[[ "${WM_PROJECT_VERSION:-}" == '13' ]] || { echo '[env] Need Foundation OpenFOAM 13' >&2; exit 2; }
# This is the child-process budget, not the aggregate PyTorch training budget.
export OMP_NUM_THREADS="${POREAI_COLLECT_THREADS_PER_CASE:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
