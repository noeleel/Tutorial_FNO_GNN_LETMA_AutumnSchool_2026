#!/bin/bash
#SBATCH --job-name=letma_ffno
#SBATCH --account=m26266
#SBATCH --partition=gpu
#SBATCH --nodelist=m-gpu[07-08]
#SBATCH --reservation=test_mig_letma
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# LETMA 2026 - validate the FFNO infrasound lab after a fno_tuto/infrasound.py
# change: executes BOTH ffno notebooks in place (solution keeps its outputs,
# student is re-stripped afterwards from the repo copy).
set -u
ROOT=/home/projects/m26266/letma2026-ml-labs
export PYTHONPATH="$ROOT"
export MPLBACKEND=Agg
unset DISPLAY
cd "$ROOT"

echo "=== node: $(hostname) | job $SLURM_JOB_ID ==="
"$ROOT/env/fno-gnn/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

for nb in lab_ffno_infrasound; do
  echo "=== executing ${nb}_solution ==="
  t0=$SECONDS
  "$ROOT/env/fno-gnn/bin/jupyter" nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1500 "notebooks/labs/${nb}_solution.ipynb" 2>&1 | tail -3
  echo "=== ${nb} solution finished in $((SECONDS - t0)) s ==="
done
echo FFNO_DONE
