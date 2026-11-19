#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/python_env.sh
echo '[02_generate] PoreAI generate'
python -m scripts.pipeline generate "$@"
