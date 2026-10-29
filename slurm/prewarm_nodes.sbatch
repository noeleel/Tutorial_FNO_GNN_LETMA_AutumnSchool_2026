#!/bin/bash
#SBATCH --job-name=letma_prewarm
#SBATCH --account=m26266
#SBATCH --partition=gpu
#SBATCH --nodelist=m-gpu[07-08]
#SBATCH --reservation=test_mig_letma
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.20gb:1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# =============================================================================
# LETMA 2026 - pre-warm the MIG nodes BEFORE the hands-on session.
#
# The first euporie launch on a cold node takes 1-2 minutes because everything
# (euporie, jupyter, torch, numpy, matplotlib, pygments, ...) is paged in from
# the shared NFS filesystem. The Linux page cache is shared per node by ALL
# users, so one warm-up run on each node makes every participant's launch take
# a few seconds instead.
#
# Run once, ~10 minutes before the session (takes ~3 min on both nodes):
#   ssh hedy
#   sbatch /home/projects/m26266/letma2026-ml-labs/slurm/prewarm_nodes.sbatch
# Watch progress with:  squeue -u $USER   (wait for the job to finish)
# =============================================================================
srun --nodes=2 --ntasks-per-node=1 \
    bash /home/projects/m26266/letma2026-ml-labs/slurm/prewarm_one_node.sh
