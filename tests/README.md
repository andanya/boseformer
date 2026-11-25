# Boseformer tests

The tests check model behaviour, numerical physics, and the training and
inference workflows used by the scripts in this repository.

## What the tests check

- `test_paper_workflow.py`: the N=25 periodic dipolar configurations for
  training, fine-tuning, single-coupling training, and inference. Short runs
  exercise KFAC and Metropolis updates, checkpoint restoration, changing the
  coupling for inference, frozen weights, energy components, and the
  condensate estimator.
- `test_periodic_physics.py`: the Ewald energy against an independent periodic
  image sum, and the full N=25 network's particle-permutation invariance,
  periodicity, and dependence on the interaction strength.
- `test_disk_configs.py`: the Gaussian pair potential and disk wall against
  analytic energies, conditional and unconditioned training, checkpoint-based
  inference, and rejection of incompatible configuration settings.
- `test_smooth_contact.py`: derivatives of the smooth periodic dipolar Jastrow
  against an independent analytic calculation near particle contact, including
  cancellation of the leading kinetic and potential energy divergences.
- `test_model_reference.py`: fixed parameter initialization, wavefunction values,
  position gradients, and local energies for disk, square, and hexagonal models;
  checkpoint restoration, resumed KFAC training, and inference.
- `test_geometry.py`: cell wrapping and symmetries for both periodic geometries,
  disk-center translation, invalid geometry rejection, and square-cell Gaussian
  training with Adam.

The fixtures in `tests/data/` contain CPU reference values and a 41 KB
checkpoint for a four-particle test model after two KFAC steps. The reference
JSON uses seed 23, four particles, one attention layer, and two product
channels in the pinned float32 environment. Fixture source:
`a87793730efe952fbe006753373fa5f0be1aef03`.

## CPU

From the repository root, install the locked uv environment. The CPU command
omits CUDA libraries; leave off `--no-group cuda` to install GPU support too.

```bash
uv sync --locked --no-group cuda
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --locked --no-group cuda python -m pytest -q tests --durations=5
```

The short CPU training runs use four particles and 12 walkers per device.
The periodicity, permutation, and coupling-conditioning checks use the full
25-particle forward pass. Checkpoints and outputs are written to pytest
temporary directories. GPU tests are skipped unless explicitly enabled.

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
