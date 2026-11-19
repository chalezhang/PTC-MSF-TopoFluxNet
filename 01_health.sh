#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/foam_env.sh
echo '[01_health] PoreAI health'
python -m scripts.pipeline health "$@"
