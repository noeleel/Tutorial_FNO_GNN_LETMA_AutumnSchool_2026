#!/bin/bash
#SBATCH --job-name=jax_fno_test
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
set -u
ROOT=/home/projects/m26266/letma2026-ml-labs
export PYTHONPATH=$ROOT
export MPLBACKEND=Agg
unset DISPLAY
cd $ROOT
echo "=== node: $(hostname) | job $SLURM_JOB_ID ==="
for v in jax_fno_solution jax_fno_student; do
  echo "=== [jax] executing $v ==="
  t0=$SECONDS
  $ROOT/env/jax-hedy/bin/jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1200 notebooks/extensions/$v.ipynb 2>&1 | tail -3
  echo "=== [jax] $v finished in $((SECONDS - t0)) s ==="
done
echo ALL_DONE
