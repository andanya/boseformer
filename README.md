# BOSENET

Neural quantum states for interacting bosons in the continuum.

**Author:** Daniil Antonenko (antonenko.daniel@gmail.com)

BoseNet uses variational Monte Carlo and a bosonic architecture derived from
[Google DeepMind's FermiNet and Psiformer](https://github.com/google-deepmind/ferminet).

## Environment

Use the `ferminet-jax` Conda environment. The Python/library versions and source
repository commits are recorded in [requirements-ferminet-jax.yml](requirements-ferminet-jax.yml).

## Simulation configuration

The physical system is selected by the config passed to `train2DEG.train(cfg)`.
Start with `ferminet.base_2DEG_config.default()` and set the following fields.

| Config field | Meaning |
| --- | --- |
| `geometry` | `'disk'` or `'periodic'`. Default `'auto'` selects periodic geometry when a feature-layer lattice is supplied, otherwise disk geometry. This keeps existing paper scripts compatible. |
| `system.electrons` | Particle counts `(N, 0)` for spin-polarized bosons; the scripts can also supply this through `system.pyscf_mol`. |
| `system.ndim` | Spatial dimension; these simulations use `2`. |
| `interaction_type` | `'inverse_cubic'` (default), with pair potential `lambda / (r + a)^3`, or `'short_range'`, with normalized Gaussian pair potential `lambda * exp(-r^2 / (2 sigma^2)) / (2 pi sigma^2)`. |
| `interaction_strength_training_set` | List of lambdas for conditional training. `None` or an empty list uses `interaction_strength` for every walker. |
| `interaction_strength` | Lambda for a single-coupling training or inference run. Also controls Gaussian interaction strength. |
| `interaction_small_length_cutoff` | Softening length `a` for inverse-cubic interactions; `0.0` gives the bare dipolar interaction. |
| `short_range_width` | Gaussian width `sigma`, default `0.1`. |
| `disk_radius`, `barrier_sharpness` | Disk wall radius (default `10`) and sharpness. The wall is centred on the first reference atom; its height is `10 * barrier_sharpness` and rim width is `0.1 / barrier_sharpness`. No wall is applied in periodic geometry. |
| `lattice_type` | Input to `base_2DEG_config.make_lattice`: `'hexagonal'` or `'square'`. Pass the resulting matrix to the feature layer; setting this string alone does not construct a cell. |
| `network.make_feature_layer_fn` | Periodic: `'ferminet.pbc.feature_layer.make_pbc_feature_layer'`. Disk: `''` for the standard open-space features. |
| `network.make_feature_layer_kwargs` | Periodic: `{'lattice': lattice, 'include_r_ae': False}`. Disk: `{}`. Lattice vectors are columns. |
| `network.make_envelope_fn` | Periodic: `'ferminet.envelopes.make_null_envelope'`. Disk: `''` for the isotropic envelope. |
| `interaction_ewald` | Exact periodic dipolar Ewald summation. The paper scripts enable it with real/reciprocal shells `4`/`12`. Requires periodic inverse-cubic physics. |
| `interaction_truncation_limit`, `interaction_cutoff_radius`, `interaction_tail_correction` | Controls for explicit image sums and optional dipolar tail correction when Ewald is off. Tail correction requires a periodic lattice and cutoff and cannot be combined with Ewald. |
| `network.smooth_periodic_jastrow` | Fixed smooth periodic dipolar Jastrow. Enabled in the paper scripts; disable for disk or Gaussian interactions. |
| `network.lambda_conditioning.mode` | `'additive'` (default) or `'film'` for a conditional network; `'none'` removes learned lambda conditioning for a single-lambda model. |
| `network.boson_head` | `'product'` (paper scripts) or `'sum'`. |
| `optim.optimizer` | `'kfac'` for training; `'none'` for frozen-weight inference. |
| `log.restore_path`, `log.save_path` | Checkpoint source and output directory. |

The model is conditional by default. Supply a list of lambdas to train across
couplings. To train or evaluate it at just one coupling, set
`interaction_strength_training_set = None` and `interaction_strength = value`.
Keep the conditioning mode unchanged when loading a conditional checkpoint.
To train a separate unconditioned model, also set
`network.lambda_conditioning.mode = 'none'`; that parameter layout differs
from conditional checkpoints. The optional fixed Jastrow still uses the
physical coupling even when learned conditioning is disabled.

## Entry points

Run from the repository root with the environment activated:

```bash
conda activate ferminet-jax
PYTHONPATH=. python scripts/dipole-pbc-hex-N25-pbc.py
```

The 25-particle periodic hexagonal bare-dipole scripts retain their original
model architecture, Ewald settings, lambda lists, and checkpoint/output paths:

- `scripts/dipole-pbc-hex-N25-pbc.py`: conditional training.
- `scripts/dipole-pbc-hex-N25-pbc-ft2040.py`: conditional fine-tuning.
- `scripts/dipole-pbc-hex-N25-pbc-int30.py`: training at lambda 30 using the same conditional architecture.
- The matching `scripts/infer-*.py` files: frozen inference, taking lambda and optionally iteration count as positional arguments.
- The `scripts/run-infer-*.sh` launchers: the corresponding Slurm inference jobs.

The retained disk example is `scripts/bosef-10e-rep01.py`, with 10 particles
(the largest disk particle count tracked on the former local main), Gaussian
repulsion, and a configurable lambda list. Run it with:

```bash
PYTHONPATH=. python scripts/bosef-10e-rep01.py
# On the configured cluster, with ferminet-jax activated:
sbatch scripts/run-disk-N10-a100.sh
```

Historical root-level simulation Python and shell scripts are kept locally
but are no longer tracked. `setup.py` remains tracked for package installation.
Training statistics are written to `train_stats.csv`; the latest walkers are
written to `walkers_last.host0.npy` in the configured output directory.

## Tests

See [tests/README.md](tests/README.md) for CPU checks and opt-in short GPU tests.

## License

Apache 2.0.
