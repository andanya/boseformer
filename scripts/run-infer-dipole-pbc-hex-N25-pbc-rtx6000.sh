#!/bin/bash
#SBATCH --job-name=inf-25e-smoothp-bare
#SBATCH --partition=gpu_rtx6000
#SBATCH --gpus=rtx_pro_6000_blackwell:1
#SBATCH --time=12:00:00
#SBATCH --mem=140G
#SBATCH --array=0-24%12
#SBATCH --no-requeue
#SBATCH --output=./run-infer-dipole-25e-pbc-gen-HEX-smoothperiodic-bare/slurm_%A_%a.out
#SBATCH --error=./run-infer-dipole-25e-pbc-gen-HEX-smoothperiodic-bare/slurm_%A_%a.err

# Run uv sync --locked in the checkout before submitting this job.
# Submit with sbatch from the repository root; Slurm runs a spool copy.
# Direct Bash runs locate the repository relative to this script.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}" || exit 1
else
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
fi
if [[ ! -f pyproject.toml || ! -d network ]]; then
  echo "Run sbatch from the repository root (containing pyproject.toml and network/)." >&2
  exit 1
fi


LAMBDAS=(0.1 0.5 1.0 2.0 5.0 10.0 15.0 20.0 22.0 24.0 26.0 28.0 30.0 31.0 32.0 33.0 34.0 35.0 36.0 38.0 40.0 45.0 50.0 65.0 80.0)
LAM=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: lambda = $LAM"

nvidia-smi | head -3
echo "GPU type: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"

# Inference: keep autotune off (fast compile, no steady-state benefit needed).
export XLA_FLAGS="--xla_gpu_autotune_level=0"

uv run --locked --no-sync python scripts/infer-dipole-pbc-hex-N25-pbc.py $LAM
