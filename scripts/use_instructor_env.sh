#!/usr/bin/env bash
# =============================================================================
# LETMA 2026 - one-line access to the school's SHARED, validated environment.
# Pre-staged on the Hedy cluster at:
#   /home/projects/m26266/letma2026-ml-labs   (repo + venvs, group-readable)
# Usage (participants, on the login node or inside a GPU job):
#   source /home/projects/m26266/letma2026-ml-labs/scripts/use_instructor_env.sh
# Then edit YOUR OWN copy of a lab (never the shared file):
#   cp /home/projects/m26266/letma2026-ml-labs/notebooks/labs/lab_fno_student.ipynb ~/my_lab.ipynb
#   euporie-notebook --kernel-name python3 ~/my_lab.ipynb    # or: jupyter lab
# The 'python3' kernel IS the validated interpreter (the shared venv's own).
# For the JAX extensions, register once: /home/projects/m26266/letma2026-ml-labs/env/jax-hedy/bin/python \
#   -m ipykernel install --user --name jax-hedy --display-name "Python (JAX Hedy)"
# The environment is READ-ONLY for participants: do not pip-install into it.
# =============================================================================
set -e
SHARED_ROOT="${LETMA_SHARED_ROOT:-/home/projects/m26266/letma2026-ml-labs}"
[ -r "$SHARED_ROOT/env/fno-gnn/bin/activate" ] || { echo "Cannot read $SHARED_ROOT/env/fno-gnn (permissions or wrong cluster?)" >&2; exit 1; }
source "$SHARED_ROOT/env/fno-gnn/bin/activate"
export PYTHONPATH="$SHARED_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "shared school env active (read-only):"
echo "  python : $(command -v python) -> $(python -V 2>&1)"
echo "  torch : $(python -c 'import torch;print(torch.__version__)')"
echo "  euporie: $(command -v euporie-notebook || echo MISSING)"
echo "  project: $SHARED_ROOT (read from \$PYTHONPATH)"
