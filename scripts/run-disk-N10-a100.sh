#!/bin/bash
#SBATCH --job-name=andanya-psiformer
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --output=./andanya-psiformer-jobs/andanya-psiformer-job_%j.out
#SBATCH --error=./andanya-psiformer-jobs/andanya-psiformer-job_%j.err


# Run uv sync --locked in the checkout before submitting this job.
cd /home/da753/bosenet || exit 1

while true; do
  echo "=== GPU MEM $(date '+%H:%M:%S') ==="
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free \
             --format=csv,noheader,nounits | \
    awk -F', ' '{printf "Total: %s MiB | Used: %s MiB | Free: %s MiB\n", $1, $2, $3}'
  sleep 300
done &
MONITOR_PID=$!

nvidia-smi | head -3
echo "GPU type: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"

if [ "$#" -eq 0 ]; then
  uv run --locked --no-sync python scripts/bosef-10e-rep01.py
else
  uv run --locked --no-sync "$@"
fi
EXIT_CODE=$?

kill $MONITOR_PID 2>/dev/null
exit $EXIT_CODE
