#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/python_env.sh
echo '[05_evaluate] PoreAI evaluate'
python -m scripts.pipeline evaluate "$@"
