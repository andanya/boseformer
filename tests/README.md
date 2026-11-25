# BoseNet refactoring checks

These tests exercise the 25-particle bare dipolar runs in `scripts/`. They
check the paper's training, fine-tuning, single-coupling reference, and
inference configurations. The reduced-size runs use the actual KFAC and
Metropolis updates, Ewald potential, and smooth periodic Jastrow. They check
checkpoint restoration, the switch to a new coupling for inference, frozen
weights, energy components, and the condensate estimator.

The disk checks also verify the Gaussian pair potential and disk wall against
analytic energies, train conditional and unconditioned models with KFAC,
restore them for frozen inference, and reject incompatible config choices.

## CPU

From the repository root, install the locked uv environment. The CPU command
omits CUDA libraries; leave off `--no-group cuda` to install GPU support too.

```bash
uv sync --locked --no-group cuda
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --locked --no-group cuda python -m pytest -q tests test/test_smooth_contact.py --durations=5
```

The CPU workflow uses four particles and 12 walkers per device. The
permutation, periodicity, and coupling-conditioning checks use the full
25-particle forward pass. All checkpoints and outputs are generated in pytest
temporary directories.

## GPU

Run from the repository root on an allocated NVIDIA GPU with the uv
environment including the default CUDA dependency group. The command runs
just the two GPU cases and caps wall time at 20 minutes, including JAX
compilation.

```bash
uv sync --locked
JAX_PLATFORMS=cuda BOSENET_RUN_GPU_TESTS=1 \
  XLA_FLAGS=--xla_gpu_autotune_level=0 \
  timeout --kill-after=30s 20m uv run --locked --no-sync python -m pytest -q tests -k gpu --durations=5
```

The GPU cases use the full 25-particle network for the forward check and
for two KFAC updates followed by one frozen inference step. They use only
12 walkers per device and two Metropolis proposals per step. Compilation
time depends on the GPU and its software stack.
