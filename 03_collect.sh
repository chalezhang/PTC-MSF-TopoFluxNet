#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/foam_env.sh
echo '[03_collect] PoreAI collect'
python -m scripts.pipeline collect "$@"
