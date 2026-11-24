#!/bin/bash
#SBATCH --job-name=inf-int30
#SBATCH --partition=gpu_rtx6000
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --time=6:00:00
#SBATCH --mem=140G
#SBATCH --array=0
#SBATCH --no-requeue
#SBATCH --output=./run-infer-int30/slurm_%A_%a.out
#SBATCH --error=./run-infer-int30/slurm_%A_%a.err
cd /home/da753/bosenet || exit 1
module load miniconda
conda activate ferminet-jax
export XLA_FLAGS="--xla_gpu_autotune_level=0"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python scripts/infer-dipole-pbc-hex-N25-pbc-int30.py 30.0
