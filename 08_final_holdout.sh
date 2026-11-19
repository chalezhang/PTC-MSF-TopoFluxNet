#!/usr/bin/env bash
# Optional: generate a new untouched OpenFOAM holdout after freezing model choices.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/foam_env.sh
python -m scripts.pipeline generate --splits final_test
python -m scripts.pipeline collect --splits final_test
python -m scripts.pipeline evaluate --splits final_test
python -m scripts.pipeline visualize
