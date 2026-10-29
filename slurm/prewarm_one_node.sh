#!/bin/bash
# LETMA 2026 - per-node warm-up body; called by prewarm_nodes.sbatch via srun,
# once on each MIG node. Must live on the shared filesystem so every node sees it.
set -u
ROOT=/home/projects/m26266/letma2026-ml-labs
export PYTHONPATH="$ROOT"
unset DISPLAY
cd "$ROOT"

echo "=== pre-warming node $(hostname) | job $SLURM_JOB_ID ==="
PY="$ROOT/env/fno-gnn/bin/python"

# 1) page in the heavy imports (this is what a cold launch pays for)
"$PY" -c "
import euporie.notebook.app, jupyter_client, prompt_toolkit
import numpy, matplotlib, pygments, torch
print('imports OK on', __import__('socket').gethostname())
"

# 2) page in ipykernel via a real (short-lived) kernel start
"$PY" - <<'PYEOF'
from jupyter_client.manager import KernelManager
km = KernelManager(kernel_name="python3")
km.start_kernel()
kc = km.client(); kc.start_channels(); kc.wait_for_ready(timeout=60)
print("kernel OK")
km.shutdown_kernel()
PYEOF

# 3) read all lab notebooks once (pages the files into the node cache)
cd "$ROOT/notebooks/labs"
"$PY" - <<'PYEOF'
import pathlib
for p in pathlib.Path(".").glob("*.ipynb"):
    p.read_bytes()
    print("read", p)
PYEOF

# 4) one real euporie launch per node (renders cells, converts equations, ~60 s)
export TERM=xterm-256color
timeout 90 script -qec \
    "$ROOT/env/fno-gnn/bin/euporie-notebook --kernel-name python3 lab_fno_student.ipynb" /dev/null \
    > /dev/null 2>&1 || true
echo "=== $(hostname) warm - participant launches will be fast ==="
