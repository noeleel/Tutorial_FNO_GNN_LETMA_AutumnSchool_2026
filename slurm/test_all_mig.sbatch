#!/bin/bash
#SBATCH --job-name=letma_tests
#SBATCH --account=m26266
#SBATCH --partition=gpu
#SBATCH --nodelist=m-gpu[07-08]
#SBATCH --reservation=test_mig_letma
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# =============================================================================
# LETMA 2026 - validation complète en UNE SEULE commande (un seul rebond SSH) :
#   ssh hedy
#   sbatch /home/projects/m26266/letma2026-ml-labs/slurm/test_all_mig.sbatch
# Tout se déroule depuis l'emplacement partagé : arbre + venvs + données.
# Sortie : letma_tests_<jobid>.out dans le dossier où sbatch est soumis.
# =============================================================================
set -u
ROOT=/home/projects/m26266/letma2026-ml-labs
export PYTHONPATH="$ROOT"
export MPLBACKEND=Agg
unset DISPLAY
cd "$ROOT"

echo "=== node: $(hostname) | job $SLURM_JOB_ID | MIG gres: $SLURM_JOB_GPUS ==="
"$ROOT/env/fno-gnn/bin/python" -c "import torch; print('torch', torch.__version__, '| devices', torch.cuda.device_count(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

for nb in lab_fno lab_ffno_infrasound lab_gnn; do
  echo "=== [torch] executing ${nb}_solution ==="
  t0=$SECONDS
  "$ROOT/env/fno-gnn/bin/jupyter" nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1500 "notebooks/labs/${nb}_solution.ipynb" 2>&1 | tail -2
  echo "=== [torch] ${nb} finished in $((SECONDS - t0)) s ==="
done

echo "=== [torch] executing scaleup_darcy_fno2d ==="
t0=$SECONDS
"$ROOT/env/fno-gnn/bin/jupyter" nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 "notebooks/extensions/scaleup_darcy_fno2d.ipynb" 2>&1 | tail -2
echo "=== [torch] scaleup finished in $((SECONDS - t0)) s ==="

echo "=== [jax] env check ==="
"$ROOT/env/jax-hedy/bin/python" -c "import jax, equinox, optax; print('jax', jax.__version__, '| equinox', equinox.__version__, '| optax', optax.__version__, '| devices', jax.devices(), '| backend', jax.default_backend())"
for nb in jax_fno jax_gnn; do
  echo "=== [jax] executing ${nb}_solution ==="
  t0=$SECONDS
  "$ROOT/env/jax-hedy/bin/jupyter" nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1200 "notebooks/extensions/${nb}_solution.ipynb" 2>&1 | tail -2
  echo "=== [jax] ${nb} finished in $((SECONDS - t0)) s ==="
done
# the student versions too: they exercise the exercise-fallback paths (reference takes over)
for nb in jax_fno jax_gnn; do
  echo "=== [jax] executing ${nb}_student (fallback paths) ==="
  t0=$SECONDS
  "$ROOT/env/jax-hedy/bin/jupyter" nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1200 "notebooks/extensions/${nb}_student.ipynb" 2>&1 | tail -2
  echo "=== [jax] ${nb} student finished in $((SECONDS - t0)) s ==="
done
echo ALL_DONE
