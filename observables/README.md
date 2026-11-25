# Observables

Run each Python script from the repository root in the installed environment.
Every command supports `--help` and writes a CSV plus separate PDF/PNG figures.
The scripts share sampling and plotting helpers in this folder and use the
installed `network` package; they do not depend on Astra or its data directories.

## Checkpoint inputs

Supply `--checkpoint`, `--config`, and `--lambda`. A checkpoint directory selects
its highest numbered `qmcjax_ckpt_*.npz` file. Configs can be the committed Python
run scripts (their training call is suppressed), or JSON with the same fields.
The architecture, geometry, and Hamiltonian must match the trained model;
parameter shapes alone cannot verify settings such as the Jastrow factor.
Only load checkpoints and Python configs from trusted sources.

Several couplings can follow `--lambda`. To compare sizes, pass several paths
after `--checkpoint`, a matching list after `--config`, and `--x-axis n_particles`.
The Bragg and occupation size-scaling plots are separate figures, without insets.
Two-dimensional maps produce one figure per checkpoint and coupling.
`--training-couplings` can list the combined training and fine-tuning couplings
for red markers; otherwise markers use the training set in the supplied config.

## Sampling and uncertainties

Weights stay fixed. `--walkers` selects distinct starting configurations from
the checkpoint, `--burn-in` discards initial Metropolis steps, `--thin` sets the
steps between saved times, and `--samples` sets the number of saved times per
chain. Proposal widths adapt only during burn-in. Increase these settings to
check equilibration and sampling convergence, especially near crystallization.
Use `JAX_PLATFORMS=cpu` to select CPU execution, or `JAX_PLATFORMS=cuda` for GPU.

Means use all retained samples, with standard errors across whole-chain means.
Local-energy variances use delete-one-chain jackknife errors. These estimates
include time correlations within each chain but assume chains are independent;
they do not measure optimization error or establish equilibration. CSV columns
record the checkpoint, coupling, chain count, sampling settings, and acceptance.
`--chunk-size` bounds network and local-energy evaluation batches.

Pair correlations use the finite-N normalization that gives one for an ideal
uncorrelated gas. Spatial correlations and momentum observables require a
periodic cell; energy observables also support disk geometry. Square and
hexagonal cells use physical nearest images. Radial pair correlations include
only complete annuli inside the cell's inradius. The 2D histogram normalizes
boundary pixels by their area inside the cell, recorded as `bin_area` in the
CSV. Pixels with no cell overlap have `inside_cell=False` and are masked.

Bragg peaks average complete sixfold stars (fourfold for a square cell) on
the integer shell `Q=N`, configurable with `--shell`. Star selection uses the
other half of the chains, then the roles are reversed. Error bars are conditional
on those selections. Radial structure factors average all allowed reciprocal
vectors of equal length, excluding zero. Feynman–Bijl uses
`omega = k**2 / (2*S)` in units `hbar = m = 1`, taking the inverse after shell
averaging; it is a single-mode estimate, with first-order propagated errors.
Occupation uses signed wavefunction ratios and arithmetic means. Its CSV
contains both `n(0)` and `n(0)/N`; `--fraction` selects the latter for plotting.

## Pure-Jastrow comparisons

The two comparison scripts optimize `R0/Lmin` for the pure smooth periodic
Jastrow state at each coupling, using the bare dipolar Ewald Hamiltonian.
`--baseline-grid` controls the scan and `--baseline-scan-samples` its length.
A local convex quadratic in log length refines an interior minimum. The lower
grid is extended once if needed; the CSV flags a remaining boundary minimum.
Production estimates use fresh random seeds, separately from the scan and
network samples. Errors are conditional on the selected length and do not
include its optimization uncertainty. These commands cost more than a single
network evaluation because they sample several reference states.

Alternatively, `--baseline-table` accepts a CSV with `n_particles`, `lambda`,
`energy_mean`, `energy_se`, `energy_variance`, `energy_variance_se`, and
`rmatch_frac`, for the same geometry, density, and Hamiltonian. The paper-table
aliases `N` and `energy_mean_se` are accepted. Each size/coupling must occur once.

## Training histories

`training_convergence.py` and `training_variance.py` take `--stats` (one or more
CSV files or run directories), `--n-particles`, and optionally `--labels`.
`--bin-size` controls averaging over consecutive logged rows. Repeated steps
keep their final entry. Training-energy shading shows temporal standard
deviation, not a confidence interval. The variance script plots logged
`ewvar/N**2`, the exponentially weighted variance of the batch-mean energy;
this is distinct from the individual local-energy variance.
