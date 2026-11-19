#!/usr/bin/env bash
# Start-to-finish fresh run; no legacy cache or inherited CFD/AI results.
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
from pathlib import Path
import shutil
root=Path.cwd().resolve()
for name in ('cases','data','models','results'):
    path=root/name
    if path.is_symlink():raise RuntimeError(f'Unsafe output symlink: {path}')
    if path.is_dir():shutil.rmtree(path)
PY
for phase in 01_health 02_generate 03_collect 04_train 05_evaluate 06_visualize 07_report; do
  echo "[RUN] $phase"
  bash "${phase}.sh"
done
