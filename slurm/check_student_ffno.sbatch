#!/bin/bash
#SBATCH --job-name=ffno_student_check
#SBATCH --account=m26266
#SBATCH --partition=gpu
#SBATCH --nodelist=m-gpu[07-08]
#SBATCH --reservation=test_mig_letma
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -u
ROOT=/home/projects/m26266/letma2026-ml-labs
export PYTHONPATH="$ROOT"; export MPLBACKEND=Agg; unset DISPLAY
cp "$ROOT/notebooks/labs/lab_ffno_infrasound_student.ipynb" ~/ffno_student_check.ipynb
cd "$ROOT"
"$ROOT/env/fno-gnn/bin/jupyter" nbconvert --to notebook --execute --allow-errors \
    --ExecutePreprocessor.timeout=1200 ~/ffno_student_check.ipynb 2>&1 | tail -2
"$ROOT/env/fno-gnn/bin/python" - <<'PYEOF'
import json
nb = json.load(open('/home/elonoele/ffno_student_check.ipynb'))
errs = [(i, o.get('ename'), o.get('evalue', '')[:80])
        for i, c in enumerate(nb['cells']) for o in c.get('outputs', []) if o.get('output_type') == 'error']
print("STUDENT_ERRORS:", len(errs))
for e in errs: print(e)
PYEOF
echo STUDENT_CHECK_DONE
