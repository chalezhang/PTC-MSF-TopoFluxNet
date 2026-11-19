#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f .venv/bin/activate ]] || { echo '[env] First run bash 00_setup.sh' >&2; exit 2; }
source .venv/bin/activate
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${POREAI_THREADS:-14}"
export OPENBLAS_NUM_THREADS="${POREAI_THREADS:-14}"
export MKL_NUM_THREADS="${POREAI_THREADS:-14}"
