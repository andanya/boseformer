#!/bin/bash
#SBATCH --job-name=inf-25e-ft2040
#SBATCH --partition=gpu_rtx6000
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --time=12:00:00
#SBATCH --mem=140G
#SBATCH --array=0-10%12
#SBATCH --no-requeue
#SBATCH --output=./run-infer-dipole-25e-pbc-gen-HEX-smoothperiodic-bare-ft2040/slurm_%A_%a.out
#SBATCH --error=./run-infer-dipole-25e-pbc-gen-HEX-smoothperiodic-bare-ft2040/slurm_%A_%a.err

cd /home/da753/bosenet || exit 1
module load miniconda
LAMBDAS=(20.0 22.0 24.0 26.0 28.0 30.0 32.0 34.0 36.0 38.0 40.0)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}
conda activate ferminet-jax
echo "Task $SLURM_ARRAY_TASK_ID: lambda = $LAM"
export XLA_FLAGS="--xla_gpu_autotune_level=0"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python scripts/infer-dipole-pbc-hex-N25-pbc-ft2040.py $LAM
