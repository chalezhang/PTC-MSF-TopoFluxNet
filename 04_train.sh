#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/python_env.sh
echo '[04_train] PoreAI train'
python -m scripts.pipeline train "$@"
