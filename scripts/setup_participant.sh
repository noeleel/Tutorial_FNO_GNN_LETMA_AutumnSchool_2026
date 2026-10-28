#!/usr/bin/env bash
# =============================================================================
# LETMA 2026 — one-time participant setup (Hedy or any Linux box with CUDA 12).
#
# Creates your OWN virtual environment with the exact validated versions
# (requirements.txt), registers the Jupyter kernel, and smoke-tests the imports.
# Nothing is shared with or taken from the instructor's home: the environment
# is yours, built from the pinned requirements of this repository.
#
# Usage (once, on the LOGIN node — 3-6 minutes):
#   git clone <the school repository>
#   cd letma2026-ml-labs
#   bash scripts/setup_participant.sh
#
# Optional arguments:
#   bash scripts/setup_participant.sh <venv-path>       # default: ~/envs/fno-gnn
#   PYTHON=/path/to/python3.12 bash scripts/setup_participant.sh
#
# Afterwards, start a lab (inside a Slurm GPU allocation, or on your laptop):
#   source ~/envs/fno-gnn/bin/activate
#   jupyter lab
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-$HOME/envs/fno-gnn}"

# --- pick a Python (>= 3.10): Hedy's module python first, then anything sane ---
if [ -z "${PYTHON:-}" ]; then
  for cand in /opt/ohpc/pub/apps/python/3.12.4/bin/python3 python3.12 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then PYTHON="$cand"; break; fi
  done
fi
[ -n "${PYTHON:-}" ] || { echo "ERROR: no python3 found" >&2; exit 1; }
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || { echo "ERROR: need Python >= 3.10 (found $("$PYTHON" -V 2>&1))" >&2; exit 1; }
echo ">>> using $("$PYTHON" -V 2>&1) at $PYTHON"

# --- build the venv from the pinned requirements ---
echo ">>> creating $VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install -q -U pip
echo ">>> installing pinned requirements (largest wheel: torch, ~2.5 GB)"
"$VENV_DIR/bin/pip" install -q -r "$REPO_ROOT/requirements.txt"

# --- register the kernel for JupyterLab/Euporie ---
"$VENV_DIR/bin/python" -m ipykernel install --user --name fno-gnn \
  --display-name "Python (fno-gnn)" >/dev/null

# --- smoke test (CPU-level: CUDA only appears on a compute node) ---
"$VENV_DIR/bin/python" - <<'EOF'
import torch, torch_geometric, matplotlib, nbformat
print(f"torch      {torch.__version__}  (cuda available here: {torch.cuda.is_available()})")
print(f"pyg        {torch_geometric.__version__}")
print(f"matplotlib {matplotlib.__version__}")
print("smoke test OK")
EOF

cat <<EOF

Done. Your environment is $VENV_DIR (kernel registered as 'fno-gnn').
  - Activate it:            source $VENV_DIR/bin/activate
  - Start a lab notebook:   jupyter lab
  - On Hedy: request a GPU first (e.g. the school's MIG allocation), then run
    jupyter from inside the job so the notebook sees the GPU.
CUDA shows 'False' on the login node by design - it becomes True on a GPU node.
EOF
