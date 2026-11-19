#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/python_env.sh
python -m scripts.pipeline report "$@"
